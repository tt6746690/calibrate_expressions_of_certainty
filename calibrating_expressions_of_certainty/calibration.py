import warnings
import os
import re
import json
import time
import pickle
import itertools
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import ot
import scipy
from tqdm import tqdm

from .utils import parse_kv_from_string, joblib_parallel_process, metrics_binary_classification, plt_kernel_matrix_one

# ignore torch load warning due to using `torch.distributions` to represent u,v
warnings.filterwarnings(
    action='ignore', 
    category=FutureWarning,
    message=re.escape("You are using `torch.load` with `weights_only=False`")
)

warnings.filterwarnings(
    action='ignore', 
    category=scipy.integrate.IntegrationWarning,
    message=re.escape("The maximum number of subdivisions")
)

warnings.filterwarnings(
    action='ignore', 
    category=pd.errors.PerformanceWarning,
    message=re.escape("dropping on a non-lexsorted multi-index without a level parameter may impact performance.")
)

warnings.filterwarnings(
    action='ignore', 
    category=RuntimeWarning,
    message=re.escape("invalid value encountered in sqrt")
)
warnings.filterwarnings(
    action='ignore', 
    category=RuntimeWarning,
    message=re.escape("The iteration is not making good progress, as measured by the")
)
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message="No positive class found in y_true, recall is set to one for all thresholds.")



def get_llm_qa_response(model_name, file_path, dcp_dist, fallback_confidence="Likely"):
    """Given `file_path` pointing to answers to qa tasks.
        e.g., 'results/gpt-4o-mini/truthfulqa_verbconf/answers_eval_correctness.jsonl'
    """

    with open(file_path, 'r') as f:
        outputs = [json.loads(line) for line in f]

    def standardize_confidence(x):
        x = x.strip(' ."')
        if x not in dcp_dist:
            print(f'Invalid confidence phrase: "{x}" -> {fallback_confidence}')
            x = fallback_confidence
            return x
        else:
            return x

    confidences = [standardize_confidence(x['confidence']) for x in outputs]
    labels = [x['is_correct'] for x in outputs]
    dists = [dcp_dist[x] for x in confidences]
    probs = [x.mean for x in dists]

    info = {}
    for k in ['u', 'y', 'u_prob', 'u_dist']:
        info[k] = {}

    info['u'][(model_name, 'pred')] = confidences
    info['y'][(model_name, 'true')] = np.array(labels)
    info['u_prob'][(model_name, 'pred')] = np.array(probs)
    info['u_dist'][(model_name, 'pred')] = dists

    eps = 0.01
    dist_fully_confident = torch.distributions.uniform.Uniform(
        low=torch.tensor(1-eps), high=torch.tensor(1-eps/10), validate_args=False)
    dist_fully_unconfident = torch.distributions.uniform.Uniform(
        low=torch.tensor(eps/10), high=torch.tensor(eps), validate_args=False)
    info['u_dist'][(model_name, 'true')] = [
        dist_fully_confident if x == 1 else dist_fully_unconfident
        for x in labels
    ]

    return info



class StringToBetaDistributionMapper:
    """implements the __getitem__ method to return a PyTorch Beta distribution based on the input string, e.g., "Beta(1,2)". """
    def __init__(self):
        # Case-insensitive regular expression to match "Beta(alpha,beta)" or "beta(alpha,beta)"
        self.pattern = re.compile(r"(?i)beta\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)")

    def __getitem__(self, key):
        match = self.pattern.match(key)
        if not match:
            raise ValueError("Invalid format. Use 'Beta(alpha,beta)' or 'beta(alpha,beta)' where alpha and beta are numbers.")
        alpha = float(match.group(1))
        beta = float(match.group(2))
        if alpha <= 0 or beta <= 0: # in case model generates invalid parameter for beta distribution.
            alpha = 1
            beta = 1
        return torch.distributions.Beta(alpha, beta)

    def __contains__(self, item):
        return bool(self.pattern.match(item))



class StringToApproxDeltaDistributionMapper:
    """Implements the __getitem__ method to return an approximate Delta distribution
    using a Uniform distribution with a very small range, based on the input string, e.g., "0.7" or ".7"."""
    
    def __init__(self, epsilon=1e-6):
        # Updated regular expression to match integers, floats with or without leading zeros
        # self.pattern = re.compile(r"^-?(\d*\.\d+|\d+\.?\d*)$")
        self.pattern = re.compile(r"^0(\.\d+)?|1(\.0+)?$")
        self.epsilon = epsilon  # Small range for the uniform distribution

    def __getitem__(self, key):
        if not self.__contains__(key):
            raise ValueError("Invalid format. Use a number (e.g., '0.7', '.7', '-2.5', or '3').")
        
        value = float(key)  # This will correctly handle '.7' as 0.7
        lower = torch.tensor(value - self.epsilon / 2)
        upper = torch.tensor(value + self.epsilon / 2)
        lower = torch.clamp(lower, min=0, max=1)
        upper = torch.clamp(upper, min=0, max=1)
        
        return torch.distributions.Uniform(lower, upper, validate_args=False) # avoid validate args to allow for logprob outside its support -> 0

    def __contains__(self, item):
        return bool(self.pattern.match(item))



def beta_cdf_scipy(dist, x):
    dtype = x.dtype
    device = x.device
    
    alpha = dist.concentration1.detach().cpu().numpy()
    beta = dist.concentration0.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    cdf_np = scipy.special.betainc(alpha, beta, x_np)

    cdf = torch.tensor(cdf_np, dtype=dtype, device=device)
    return cdf

torch.distributions.Beta.cdf = beta_cdf_scipy


def interpolate_2_betas(dist0, dist1, lam):
    assert(0<=lam<=1)
    alpha0 = dist0.concentration1
    beta0 = dist0.concentration0
    alpha1 = dist1.concentration1
    beta1 = dist1.concentration0
    alpha = (1-lam)*alpha0 + lam*alpha1
    beta = (1-lam)*beta0 + lam*beta1
    return torch.distributions.Beta(alpha, beta)

def beta_mean(dist):
    # E{X]=\alpha / (\alpha+\beta)
    alpha = dist.concentration1
    beta = dist.concentration0
    mean = alpha / (alpha + beta)
    return mean.item()

def interpolate_betas(x, beta_list):
    import bisect
    beta_means = [beta_mean(x) for x in beta_list]
    
    i = bisect.bisect_left(beta_means, x)
    if i == 0:
        i = 1
        x = beta_means[0]
    elif i > len(beta_means)-1:
        i = len(beta_means)-1
        x = beta_means[-1]
    dist0 = beta_list[i-1]
    dist1 = beta_list[i]
    lam = (x-beta_means[i-1]) / (beta_means[i]-beta_means[i-1])
    dist_interp = interpolate_2_betas(dist0, dist1, lam)

    return dist_interp



def get_certainty_range_distribution(keys, eps=1e-6):
    certainty_ranges = [
        ("absent", (float(0), 0+eps)),
        ("very low likelihood", (0+eps, 0.05)), 
        ("low probability", (0.05, 0.25)), 
        ("intermediate probability", (0.25, 0.75)), 
        ("high probability", (0.75, 0.9)), 
        ("very high probability", (0.9, 1-eps)), 
        ("100% confident", (1-eps, float(1))),
    ]
    if isinstance(keys, tuple):
        keys = list(keys)
    if not isinstance(keys, list):
        keys = [keys]
    if isinstance(keys[0], str):
        names = [x[0] for x in certainty_ranges]
        keys = [names.index(k) for k in keys]
    prob_range = []
    low = []
    high = []
    for k in keys:
        r, (l, h) = certainty_ranges[k]
        prob_range.append(r)
        low.append(l)
        high.append(h)
    return prob_range, low, high


## Taken from https://rad.bwh.harvard.edu/diagnostic-certainty-scale/
#
certainty_range_proportions = [
    ['compatible with', 0.0, 0.0, 0.0, 0.007, 0.007, 0.042, 0.091],
    ['consistent with', 0.0, 0.0, 0.0, 0.007, 0.07, 0.113, 0.451],
    ['diagnostic of', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.38],
    ['highly suggestive of', 0.0, 0.0, 0.0, 0.0, 0.169, 0.465, 0.028],
    ['likely', 0.0, 0.0, 0.0, 0.077, 0.218, 0.056, 0.0],
    ['maybe', 0.0, 0.0, 0.035, 0.007, 0.0, 0.0, 0.0],
    ['may represent', 0.0, 0.014, 0.12, 0.507, 0.028, 0.0, 0.0],
    ['most likely', 0.0, 0.0, 0.0, 0.028, 0.31, 0.204, 0.0],
    ['possibly', 0.0, 0.049, 0.176, 0.169, 0.007, 0.0, 0.0],
    ['probably', 0.0, 0.007, 0.007, 0.028, 0.056, 0.021, 0.0],
    ['question of', 0.0, 0.042, 0.056, 0.0, 0.0, 0.0, 0.0],
    ['suggestive of', 0.0, 0.0, 0.014, 0.113, 0.056, 0.028, 0.0],
    ['suspicious for', 0.0, 0.0, 0.014, 0.028, 0.049, 0.014, 0.0],
    ['unlikely', 0.0, 0.38, 0.471, 0.0, 0.0, 0.0, 0.0],
    ['very unlikely', 0.0, 0.401, 0.007, 0.0, 0.0, 0.0, 0.0],
    ['worrisome for', 0.0, 0.0, 0.0, 0.014, 0.014, 0.014, 0.0],
    ['other', 0.0, 0.106, 0.099, 0.014, 0.014, 0.042, 0.049],
    ['absent', 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ['present', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ## additional ones that are duplicates of previous with different keywords
    ['may', 0.0, 0.014, 0.12, 0.507, 0.028, 0.0, 0.0], # same as "may represent"
]



def fit_beta_distribution(initial_dist_or_samples, num_samples=1000, strategy='mm'):
    """Fit beta distribution given `initial_dist_or_samples`."""
    
    if isinstance(initial_dist_or_samples, torch.distributions.Distribution):
        samples = initial_dist_or_samples.sample((num_samples,)).double()
    else:
        samples = initial_dist_or_samples.squeeze().double()
    eps = 1e-8
    samples = torch.clamp(samples, eps, 1-eps)

    if strategy == 'mm':
        alpha, beta, loc, scale = scipy.stats.beta.fit(samples, method='MM', floc=0, fscale=1)
    elif strategy == 'mle':
        try:
            alpha, beta, loc, scale = scipy.stats.fit(samples, floc=0, fscale=1)
        except: # fall back of method of moments, due to the delta distribution, e.g., "absent"
            alpha, beta, loc, scale = scipy.stats.fit(samples, method='MM', floc=0, fscale=1)
    else:
        raise ValueError(f'Invalid strategy = {strategy}')
    fitted_dist = torch.distributions.Beta(alpha, beta)

    return fitted_dist


def get_dcp_dist_rad(distribution_type='beta_fit_mm'):
    """get distribution for each DCP
    
        `distribution_type` determines type of distribution
            "mixture": mixture of uniform read from survey results
            "beta_fit_mm": fit beta distribution to the mixture using method of moments.

        ```
         {k: (np.round(v.concentration1.item(),decimals=1), np.round(v.concentration0.item(),decimals=1)) for k,v in dcp_dist.items() if k in ['consistent with', 'may represent', 'likely', 'possibly']}
        ```
    """
    dcp_dist = []
    for i, l in enumerate(certainty_range_proportions):
        dcp = l[0]
        mix = np.array(l[1:])
        mix = mix / np.sum(mix)
        range_inds, mix = list(zip(*[x for x in zip(np.arange(len(mix)), mix) if x[1]!=0.]))
        mix = torch.distributions.categorical.Categorical(probs=torch.tensor(mix))
        _, low, high = get_certainty_range_distribution(range_inds)
        comp = torch.distributions.uniform.Uniform(low=torch.tensor(low), high=torch.tensor(high), validate_args=False)
        dist = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp, validate_args=False)
        if distribution_type == 'mixture':
            pass
        elif distribution_type.startswith('beta_fit'):
            strategy = distribution_type[len('beta_fit_'):]
            dist = fit_beta_distribution(dist, num_samples=1000, strategy=strategy)
            ## manually set these ver peaky beta distributions because fitting beta for this two case is inaccurate.
            if dcp == 'absent':
                dist = torch.distributions.Beta(1, 10_000)
            elif dcp == 'present':
                dist = torch.distributions.Beta(10_000, 1)
        else:
            raise ValueError(f'Invalid distribution_type={distribution_type}.')
        dcp_dist.append((dcp, dist))
    return dict(dcp_dist)


def get_dcp_dist_twitter(distribution_type='beta_fit_mm'):

    cur_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(cur_path, 'perception_of_probability_word_survey.csv'))
    df = df.drop(columns=['Your Highest Level of Education',
                    'Your Gender Identity',
                    'Age Range'])
    df.columns = [x.strip('"') for x in df.columns]
    df = df / 100 # map to [0,1]

    d = df.to_dict(orient='list')
    d = {k: torch.tensor(v) for k,v in d.items()}

    dcp_dist = {}
    for dcp, samples in d.items():
        if distribution_type == 'mixture':
            raise ValueError('Cannot fit mixture of uniform distribution here.')
        elif distribution_type.startswith('beta_fit'):
            strategy = distribution_type[len('beta_fit_'):]
            dist = fit_beta_distribution(samples, strategy='mm')
        else:
            raise ValueError(f'Invalid distribution_type={distribution_type}.')
        dcp_dist[dcp] = dist

    # sort by increasing mean.
    dcp_dist = dict(sorted(dcp_dist.items(), key=lambda x: beta_mean(x[1])))

    return dcp_dist


def get_dcp_dist_prompt_llm(source):

    if source == 'prompt_gpt-4o_size=4':
        s = """
        - "Unlikely", 2, 8
        - "Maybe", 2, 2
        - "Highly Likely", 8, 2
        - "Almost Certain", 9, 1
        """
    elif source == 'prompt_gpt-4o_size=8':
        s = """
        1. "Definitely", 9, 1
        2. "Probably", 7, 3
        3. "Likely", 6, 4
        4. "Unlikely", 3, 7
        5. "Highly Unlikely", 2, 8
        6. "Maybe", 5, 5
        7. "Almost Certain", 8, 2
        8. "Doubtful", 4, 6
        """
    elif source == 'prompt_gpt-4o_size=12':
        s = """
        1. "Definitely", 9, 1
        2. "Almost certainly", 8, 2
        3. "Highly likely", 7, 3
        4. "Probably", 6, 4
        5. "Likely", 5, 5
        6. "Possibly", 4, 6
        7. "Maybe", 3, 7
        8. "Unlikely", 2, 8
        9. "Highly unlikely", 1, 9
        10. "Almost impossible", 1, 10
        11. "Certainly not", 1, 11
        12. "Impossible", 1, 12
        """
    elif source == 'prompt_gpt-4o_size=16':
        s = """
        1. "Definitely", 9, 1
        2. "Certainly", 8, 1
        3. "Highly Likely", 7, 2
        4. "Very Likely", 6, 2
        5. "Likely", 5, 3
        6. "Probably", 4, 3
        7. "Possibly", 3, 4
        8. "Maybe", 2, 5
        9. "Unlikely", 2, 6
        10. "Very Unlikely", 1, 7
        11. "Highly Unlikely", 1, 8
        12. "Almost Impossible", 1, 9
        13. "Impossible", 1, 10
        14. "Doubtful", 2, 7
        15. "Uncertain", 3, 5
        16. "Not Sure", 3, 6
        """
    elif source == 'prompt_gpt-4o_size=20':
        s = """
        1. "Definitely", 9, 1
        2. "Certainly", 8, 1
        3. "Absolutely", 9, 1
        4. "Highly likely", 7, 2
        5. "Very likely", 6, 2
        6. "Likely", 5, 2
        7. "Probably", 4, 2
        8. "Possibly", 3, 3
        9. "Maybe", 2, 2
        10. "Unlikely", 2, 5
        11. "Very unlikely", 1, 6
        12. "Highly unlikely", 1, 7
        13. "Doubtful", 1, 8
        14. "Improbable", 1, 9
        15. "Almost certain", 8, 2
        16. "Almost impossible", 1, 9
        17. "Chances are", 4, 3
        18. "Could be", 3, 4
        19. "Not sure", 2, 3
        20. "Uncertain", 2, 4
        """
    elif source == 'prompt_gemini-1.5-pro_size=12':
        s = """
        - "Certainly", 99, 1
        - "Almost Certainly", 95, 5
        - "Highly Likely", 90, 10
        - "Very Likely", 85, 15
        - "Likely", 75, 25
        - "Probably", 65, 35
        - "Maybe", 50, 50
        - "Possibly", 35, 65
        - "Unlikely", 25, 75
        - "Very Unlikely", 15, 85
        - "Highly Unlikely", 10, 90
        - "Almost Certainly Not", 5, 95
        """
    elif source == 'prompt_claude-3.5-sonnet_size=12':
        s = """
        - "Certainly", 50.0, 1.0
        - "Very likely", 20.0, 2.0
        - "Probably", 8.0, 2.0
        - "Likely", 7.0, 3.0
        - "Possibly", 5.0, 5.0
        - "Maybe", 1.0, 1.0
        - "Uncertain", 2.0, 5.0
        - "Unlikely", 2.0, 8.0
        - "Doubtful", 1.0, 10.0
        - "Very unlikely", 1.0, 20.0
        - "Almost certainly not", 1.0, 50.0
        - "Impossible", 0.1, 100.0
        """
    elif source == 'prompt_gpt-4o-mini_size=8':
        s = """
        - "Definitely", 10, 1  
        - "Very Likely", 7, 3  
        - "Likely", 5, 5  
        - "Possible", 4, 6  
        - "Maybe", 3, 7  
        - "Unlikely", 2, 8  
        - "Very Unlikely", 1, 9  
        - "Definitely Not", 1, 10  
        """
    else:
        raise ValueError(f'Invalid source: {source}')
    
    s = s.strip()

    dcp_dist = {}
    for x in s.split('\n'):
        x = x.split(',')
        dcp, alpha, beta = x
        dcp = dcp.strip()
        if dcp.startswith('-'):
            dcp = dcp.strip('- ')
        else:
            dcp = re.findall(r'\d+\.\s*(.*)', dcp)[0]
        dcp = dcp.strip('"')
        alpha, beta = float(alpha), float(beta)
        dist = torch.distributions.Beta(alpha, beta)
        dcp_dist[dcp] = dist

    dcp_dist = dict(sorted(dcp_dist.items(), key=lambda x: beta_mean(x[1])))

    return dcp_dist



def get_dcp_dist(source, distribution_type='beta_fit_mm'):
    if source == 'radiology':
        return get_dcp_dist_rad(distribution_type)
    elif source == 'social_media_poll':
        return get_dcp_dist_twitter(distribution_type)
    elif source.startswith('prompt_'):
        return get_dcp_dist_prompt_llm(source)
    elif source == 'unspecified_beta':
        return StringToBetaDistributionMapper()
    elif source == 'unspecified_delta':
        return StringToApproxDeltaDistributionMapper()
    else:
        raise ValueError(f"Invalid source: {source}")


def get_certainty_phrases(source):
    """Fetch certainty phrases used when prompting LLMs to emit confidence in natural language. 
    
        `source`
            - social_media_poll
            - prompt_gpt-4o_size=8
            - prompt_gpt-4o-mini_size=8
    """
    if source == 'social_media_poll':
        try:
            cur_file_path = __file__
        except:
            cur_file_path = '.'
        df = pd.read_csv(os.path.join(
                os.path.dirname(os.path.realpath(cur_file_path)), 
                'perception_of_probability_word_survey.csv'
            )
        )
        certainty_phrases = [x.strip('"') for x in df.columns if '"' in x]
    elif source.startswith('prompt'):
        dcp_dist = get_dcp_dist_prompt_llm(source)
        certainty_phrases = list(dcp_dist.keys())
    elif source == 'unspecified_beta':
        certainty_phrases = ['']
    elif source == 'unspecified_delta':
        certainty_phrases = ['']
    else:
        raise ValueError(f'Invalid source: {source}')
    return certainty_phrases


def plt_dcp_distribution_density(dcp_dist, ncols=8):

    dcp_dist = dict(sorted(dcp_dist.items(), key=lambda x: x[1].mean.item()))
    
    plt_params = {
        'font.size': 30,
        'font.family': 'Times New Roman',
        "legend.frameon": False,
        "axes.titlepad": 10,
        "lines.linewidth": 5,
    }

    color = '#99D6FF'

    with plt.rc_context(plt_params):
        nrows = len(dcp_dist) // ncols + int(len(dcp_dist) % ncols > 0)
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols,4*nrows), sharex=True, sharey=True)

        xs = np.linspace(0,1,200)
        for i, (dcp, dist) in enumerate(dcp_dist.items()):
            ax = axs.reshape(-1)[i]
            ys = dist.log_prob(torch.tensor(xs)).exp().numpy()
            ax.plot(xs, ys, color=color)
            ax.fill_between(xs, ys, 0, color=color, alpha=0.3)

            fontsize = None if len(dcp) < 15 else 25
            ax.text(
                0.5, .9,             # x and y coordinates
                # f"\"{dcp}\"",                   # The text to display
                f"“{dcp}”",                   # The text to display
                color='k', 
                ha='center',           # horizontal alignment
                transform=ax.transAxes, # Coordinate system: relative to the axes
                fontsize=fontsize,
            )
            ax.set_ylim((0,5))
    
        for i in range(nrows):
            # axs.reshape(nrows, ncols)[i, 0].set_ylabel('$Density$')
            axs.reshape(nrows, ncols)[i, 0].set_yticks([0,2,4])
        for j in range(ncols):
            # axs.reshape(nrows, ncols)[-1, j].set_xlabel('$x$')
            axs.reshape(nrows, ncols)[-1, j].set_xticks([0, 1])
    
        fig.tight_layout(pad=.05)

    return fig


def plt_dcp_usage(L, names=None):

    if names is None:
        names = sorted(list(set(L)))

    # Count occurrences of each string in the list
    counter = Counter(L)

    # Calculate fractions (frequency)
    total_count = len(L)
    fractions = {item: counter[item] / total_count for item in counter}

    # Ensure all names are included with fraction 0 if not present
    for name in names:
        if name not in fractions:
            fractions[name] = 0.0

    # Get fractions in the order of `names`
    ordered_fractions = [fractions[name] for name in names]

    # Create the bar plot
    fig,ax=plt.subplots(1, 1, figsize=(6, 4))

    ax.bar(names, ordered_fractions, color='skyblue')
    ax.set_xlabel('Certainty Phrases')
    ax.set_ylabel('Fraction of Occurrence')

    ax.set_xticklabels(names, rotation=-45, ha='left')

    return fig, ax



def torch_distributions_expected_value(d, fn, l, r):
    """Take expectation of a `fn` with respect to distribution `d` over interval `(l,r)`.
            - handles mixture distribution
            - intersect (l,r) with distribution's support to make integration more accurate.
    """

    if isinstance(d, torch.distributions.mixture_same_family.MixtureSameFamily):
        if isinstance(d.component_distribution, torch.distributions.uniform.Uniform):
            low = d.component_distribution.low
            high = d.component_distribution.high
            component_distributions = []
            for a, b in zip(low, high):
                component_distributions.append(
                    torch.distributions.uniform.Uniform(low=a, high=b, validate_args=False))
        else:
            raise ValueError(f'[torch_distributions_expected_value] Have not implemented for component distribution: {type(d.component_distribution)}')
        Efns = [torch_distributions_expected_value(comp_d, fn, l, r) for comp_d in component_distributions]
        Efn = torch.sum(torch.tensor(Efns) * d.mixture_distribution.probs, dim=-1)
        Efn = Efn.item()
        return Efn
    else:
        if isinstance(d, torch.distributions.uniform.Uniform):
            # update (l,r) only on distribution's support
            support = d.support
            support = [support.lower_bound.item(), support.upper_bound.item()]
            l = max(l, support[0])
            r = min(r, support[1])

        integrand_fn = lambda x: fn(x) * d.log_prob(torch.tensor([x])).exp().numpy()
        Efn = scipy.integrate.quad(integrand_fn, l, r, limit=3)[0]
    return Efn


def compute_calibration_over_dist(u, v, num_bins=10, calibration_type=1, integrate_dist='none'):
    """Same as `compute_calibration` but now each example's prediction and label 
        are `torch.distributions.Distribution` instead of probability values. """

    assert(len(u) == len(v))
    assert(num_bins > 0)
    assert(isinstance(calibration_type, int))

    num_samples = len(u)
    bins = np.linspace(0., 1., num_bins+1)

    def get_statistics_u_quadrature(dist_list, bins, num_bins):
        """Computes P(U\in I_m) and E[U1[U\in I_m]] with quadrature"""
        P_U_in_bins = []
        E_U_in_bin = []
        for d in dist_list:
            F_u_in_bins = d.cdf(torch.tensor(bins)).numpy() # F_U(u <= b) for b in bins
            P_U_in_bins.append(F_u_in_bins[1:] - F_u_in_bins[:-1]) # P(U in I_m) for m in M
            E_U_in_bin.append([
                torch_distributions_expected_value(
                    d=d, 
                    fn=lambda x: x,
                    l=bins[m],
                    r=bins[m+1],
                ) for m in range(num_bins)
            ])
        P_U_in_bins = np.array(P_U_in_bins)
        E_U_in_bin = np.array(E_U_in_bin)
        return P_U_in_bins, E_U_in_bin

    def get_statistics_u_pdf(dist_list, bins, num_bins):
        """Computes P(U\in I_m) and E[U1[U\in I_m]] with f_U(I_m)*\delta approximation. """
        bin_width = 1/num_bins
        bins_center = bins + bin_width/2
        bins_center = bins_center[:-1]
        f_U_at_bin_center = []
        for d in dist_list:
            f_U_at_bin_center.append(d.log_prob(torch.tensor(bins_center)).exp().numpy())
        # (num_unique_u, num_bins)
        f_U_at_bin_center = np.array(f_U_at_bin_center)
        P_U_in_bins = f_U_at_bin_center * bin_width
        E_U_in_bin = bins_center * bin_width * f_U_at_bin_center
        return P_U_in_bins, E_U_in_bin

    ## compute per-bin indicator, confidence, and counts
    u_unique = list(set(u))

    if integrate_dist == 'all':
        P_U_in_bins, E_U_in_bin = get_statistics_u_quadrature(u_unique, bins, num_bins)
    elif integrate_dist == 'none': # misses delta distribution
        P_U_in_bins, E_U_in_bin = get_statistics_u_pdf(u_unique, bins, num_bins)
    elif integrate_dist == 'endbin':
        P_U_in_bins0, E_U_in_bin0 = get_statistics_u_quadrature(u_unique, bins[[0,1]], 1)
        P_U_in_bins1, E_U_in_bin1 = get_statistics_u_pdf(u_unique, bins[1:-1], num_bins)
        P_U_in_bins2, E_U_in_bin2 = get_statistics_u_quadrature(u_unique, bins[[-2,-1]], 1)
        P_U_in_bins = np.hstack((P_U_in_bins0, P_U_in_bins1, P_U_in_bins2))
        E_U_in_bin = np.hstack((E_U_in_bin0, E_U_in_bin1, E_U_in_bin2))
    elif isinstance(integrate_dist, list):
        u_use_quad = [x for x in u_unique if x in integrate_dist]
        u_use_pdf = [x for x in u_unique if x not in integrate_dist]
        u_unique = u_use_quad + u_use_pdf
        assert(len(u_use_quad)+len(u_use_pdf) == len(u_unique))
        P_U_in_bins0, E_U_in_bin0 = get_statistics_u_quadrature(u_use_quad, bins, num_bins)
        P_U_in_bins1, E_U_in_bin1 = get_statistics_u_pdf(u_use_pdf, bins, num_bins)
        P_U_in_bins = np.vstack((P_U_in_bins0, P_U_in_bins1)) # (#unique u, num_bins)
        E_U_in_bin = np.vstack((E_U_in_bin0, E_U_in_bin1)) # (#unique u, num_bins)
    else:
        raise ValueError(f"Invalid integrate_dist {integrate_dist}")

    prob_y_positive = []
    v_unique = list(set(v))
    for d in v_unique:
        prob_y_positive.append(
            (d.cdf(torch.tensor([1.])) - d.cdf(torch.tensor([0.5]))).numpy()
        )
    prob_y_positive = np.array(prob_y_positive)

    # (N, num_bins)
    inds = np.array([u_unique.index(d) for d in u])
    P_U_in_bins = P_U_in_bins[inds]
    E_U_in_bin = E_U_in_bin[inds]

    inds = np.array([v_unique.index(d) for d in v])
    # (N,)
    prob_y_positive = prob_y_positive[inds]

    ## combine to estimate ECE
    eps = 1e-8
    bin_counts = P_U_in_bins.sum(axis=0)
    bin_indicators = (prob_y_positive * P_U_in_bins).sum(axis=0) / (bin_counts + eps)
    bin_confidences = E_U_in_bin.sum(axis=0) / (bin_counts + eps)
    # assert(np.allclose(np.sum(bin_counts), num_samples))

    avg_acc = np.sum(bin_indicators * bin_counts) / (np.sum(bin_counts) + 1e-8)
    avg_conf = np.sum(bin_confidences * bin_counts) / (np.sum(bin_counts) + 1e-8)

    gaps = np.abs(bin_indicators - bin_confidences)
    ece = np.sum(gaps * bin_counts) / (np.sum(bin_counts) + 1e-8)
    mce = np.max(gaps)
    
    return { "accuracies": bin_indicators, 
             "confidences": bin_confidences, 
             "counts": bin_counts, 
             "bins": bins,
             'gaps': gaps,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "ece": ece,
             "mce": mce }


def compute_calibration(labels, scores, num_bins=10, calibration_type='confidence'):
    """Collects predictions into bins used to draw a reliability diagram. 

        labels, (N, )
            ground-truth labels in {1, ..., K}
        scores, (N, ) or (N, K)
            predicted probability
        calibration_type
            'confidence': computes confidence calibration
            1, ..., K: compute classwise caliberation

        Modified from https://github.com/hollance/reliability-diagrams/tree/master
    """
    if isinstance(labels, (list, tuple)):
        labels = np.array(labels)
    if isinstance(scores, (list, tuple)):
        scores = np.array(scores)
    assert(len(scores) == len(labels))
    assert(num_bins > 0)
    assert(np.all(scores >= 0))

    num_samples = len(labels)
    
    if scores.ndim == 1:
        # convert [p, ...] to [(1-p, p), ...] assuming p = P̂(Y=+|X)
        scores_onehot = np.stack([1-scores, scores]).T
        pred = np.argmax(scores_onehot, axis=1) # predicted labels

    if calibration_type == 'confidence':
        indicators = (labels == pred) # accuracy
        confidences = np.max(scores_onehot, axis=1)
    elif isinstance(calibration_type, int):
        k = calibration_type
        indicators = (labels == k) # k-th class membership
        confidences = scores_onehot[:,k]
    else:
        raise ValueError(f'calibration_type={calibration_type} not implemented.')
        
    ## group samples into bins by scores/confidence
    bins = np.linspace(0., 1., num_bins+1)
    indices = np.digitize(confidences, bins, right=True) # note the first bucket is (-\infty, bins[0]]
    # wpq: interval left-open, therefore if confidence=0, then will fall into 0-th, instead of 1-st bucket
    # therefore, add the counts back to the first bucket
    assert(np.max(indices) <= num_bins)
    indices[indices==0] = 1
    indices -= 1 # convert indices to be 0-indexed

    bin_indicators = np.zeros(num_bins, dtype=np.float64)
    bin_confidences = np.zeros(num_bins, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=np.int64)
    for b in range(num_bins):
        selected = np.where(indices == b)[0]
        if len(selected) > 0:
            bin_indicators[b] = np.mean(indicators[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)
    assert(np.sum(bin_counts) == num_samples)

    ## compute metrics
    avg_acc = np.sum(bin_indicators * bin_counts) / num_samples
    avg_conf = np.sum(bin_confidences * bin_counts) / num_samples

    gaps = np.abs(bin_indicators - bin_confidences)
    ece = np.sum(gaps * bin_counts) / num_samples
    mce = np.max(gaps)

    return { "accuracies": bin_indicators, 
             "confidences": bin_confidences, 
             "counts": bin_counts, 
             "bins": bins,
             'gaps': gaps,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "ece": ece,
             "mce": mce }


def estimate_confidence_interval_with_bootstrap(estimator, data, num_samples=200, alpha=0.05, n_jobs='auto', data_getitem_fn=None):
    """Assume `estimator` returns a dictionary of statistics, 
        apply bootstrap with replacement for `num_samples` repetitions. 
        Return `alpha`-confidence interval. 
        
        `data_getitem_fn`
            (data, index_example, index_bootstrap_sample)
    """
    np.random.seed(0)

    N = len(data)
    bootstrap_indices_list = [np.random.choice(N, size=N, replace=True) for _ in range(num_samples)]
    if data_getitem_fn is None:
        data_list = [[data[i] for i in inds] for inds in bootstrap_indices_list]
    else:
        data_list = [[data_getitem_fn(data, i, j) for i in inds] for j, inds in enumerate(bootstrap_indices_list)]

    if n_jobs == 'auto':
        from sys import platform
        if platform == 'darwin':
            n_jobs = 1 if num_samples < 15 else 8
        else:
            n_jobs = 1 if num_samples < 25 else 25

    outputs_list = joblib_parallel_process(
        fn=estimator,
        iterable=data_list,
        n_jobs=n_jobs,
        use_tqdm=False,
    )
    outputs = defaultdict(list)
    for d in outputs_list:
        for k, v in d.items():
            outputs[k].append(v)
            
    # [(N,), ...] -> (num_bootstrap_samples, N)
    outputs = {k: np.stack(v) for k, v in outputs.items()}

    def compute_confidence_interval(estimates):
        # `estimates` (num_bootstrap_samples, ...)
        low = np.percentile(estimates, 100*(alpha/2), axis=0)
        mid = np.percentile(estimates, 100*(.5), axis=0)
        high = np.percentile(estimates, 100*(1 - alpha/2), axis=0)
        return low, mid, high

    for k in list(outputs.keys()):
        outputs[f'{k}_low'], outputs[k], outputs[f'{k}_high'] = compute_confidence_interval(outputs[k])

    return outputs



def compute_calibration_with_bootstrap(
        pathology,
        calibrate_over_dist=True,
        calibration_type=1,
        num_bins=10,
        bootstrap_num_samples=1,
        bootstrap_alpha=0.05,
        include_calibration_metrics=True,
        include_cls_metrics=False,
        n_jobs='auto',
        **kwargs,
    ):
    """Compute calibration using bootstrap. 

        `pathology`
        `calibrate_over_dist`: bool
            set to `True` to compute calibration over distribution
            set to `False` to compute calibration over probability
        `calibration_type`: int or str
            'confidence': computes confidence calibration
            1, ..., K: compute classwise caliberation
        `num_bins`: int
            number of bins that partition the interval [0,1]
        `bootstrap_num_samples`: int
            number of bootstrap samples
        `bootstrap_alpha`: float
            confidence level for the bootstrap confidence interval
        `include_calibration_metrics`: bool
            set to `True` to include calibration metrics
        `n_jobs`: int or str
            number of parallel jobs to run
            set to 'auto' to use all available cores
        `**kwargs`: additional arguments
            - `y`: true labels
            - `u_prob`: predicted probabilities
            - `u_dist`: predicted distributions
                - (pathology, 'pred'): predicted distribution
                - (pathology, 'true'): true distribution
            when `calibrate_over_dist` is `True`, `u_dist` is required and `y` and `u_prob` are ignored.
            when `calibrate_over_dist` is `False`, `y` and `u_prob` are required and `u_dist` is ignored.
    """

    outputs = {}

    if calibrate_over_dist:
        if include_calibration_metrics:
            u_dist = kwargs.get('u_dist', None)
            integrate_dist = kwargs.get('integrate_dist', 'all')
            def estimator(data):
                u, v = list(zip(*data))
                return compute_calibration_over_dist(
                    u=u,
                    v=v,
                    num_bins=num_bins,
                    calibration_type=calibration_type,
                    integrate_dist=integrate_dist,
                )
            data = list(zip(u_dist[(pathology, 'pred')], u_dist[(pathology, 'true')]))
            bin_data = estimate_confidence_interval_with_bootstrap(
                estimator, data, num_samples=bootstrap_num_samples, alpha=bootstrap_alpha, n_jobs=n_jobs)
            outputs.update(bin_data)

        if include_cls_metrics:
            cls_metrics = compute_classification_metrics_with_bootstrap(
                u=u_dist[(pathology, 'pred')],
                v=u_dist[(pathology, 'true')],
                bootstrap_num_samples=bootstrap_num_samples,
                bootstrap_alpha=bootstrap_alpha,
            )
            outputs.update(cls_metrics)
    else:
        if include_calibration_metrics:
            y = kwargs.get('y', None)
            u_prob = kwargs.get('u_prob', None)
            def estimator(data):
                labels, scores = list(zip(*data))
                return compute_calibration(
                    labels=labels,
                    scores=scores,
                    num_bins=num_bins,
                    calibration_type=calibration_type,
                )
            data = list(zip(y[(pathology, 'true')], u_prob[(pathology, 'pred')]))
            bin_data = estimate_confidence_interval_with_bootstrap(
                estimator, data, num_samples=bootstrap_num_samples, alpha=bootstrap_alpha, n_jobs=n_jobs)
            outputs.update(bin_data)

        if include_cls_metrics:
            cls_metrics = compute_classification_metrics_with_bootstrap(
                u=u_prob[(pathology, 'pred')],
                v=y[(pathology, 'true')],
                bootstrap_num_samples=bootstrap_num_samples,
                bootstrap_alpha=bootstrap_alpha,
            )
            outputs.update(cls_metrics)
    
    return outputs


def compute_classification_metrics_with_bootstrap(
        u,
        v,
        bootstrap_num_samples=100,
        bootstrap_alpha=0.05,
    ):
    """Note `u` and `v` could be a list of torch distributions, 
        or simply numbers representing probabilities & labels respectively. """

    def convert_dist_to_samples(X):
        if isinstance(X[0], torch.distributions.distribution.Distribution):
            X = [x.sample((bootstrap_num_samples,)) for x in X]
            X = torch.stack(X)
            X = X.numpy()
        elif isinstance(X, np.ndarray):
            assert(X.ndim ==1)
            X = np.repeat(X[:, np.newaxis], bootstrap_num_samples, axis=1)
        else:
            raise ValueError(f'Invalid type for {X}')
        return X
    
    # (N, num_samples)
    scores = convert_dist_to_samples(u)
    labels = convert_dist_to_samples(v)
    labels = (labels > .5).astype(np.int32)

    # [( scores_for_example_i, labels_for_example_i ), ...]
    data = [(scores[i,:], labels[i,:]) for i in range(len(scores))]

    def data_getitem_fn(data, ind_example, ind_sample):
        score, label = data[ind_example]
        return (score[ind_sample], label[ind_sample])
        
    def estimator(data):
        score, label = list(zip(*data))
        score = np.array(list(score))
        label = np.array(list(label))
        return metrics_binary_classification(label, score, threshold=.5)

    cls_metrics = estimate_confidence_interval_with_bootstrap(
        estimator, data, num_samples=bootstrap_num_samples, alpha=bootstrap_alpha, n_jobs=1, data_getitem_fn=data_getitem_fn)
    
    return cls_metrics



def compute_ece_from_bin_data(bin_data, exclude_endbin=True):
    bin_indicators = bin_data['accuracies']
    bin_confidences = bin_data['confidences']
    bin_counts = bin_data['counts']
    if exclude_endbin:
        bin_indicators = bin_indicators[1:-1]
        bin_confidences = bin_confidences[1:-1]
        bin_counts = bin_counts[1:-1]
    gaps = np.abs(bin_indicators - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    return ece


def plt_reliability_diagram_subplot(ax, bin_data, 
                                    draw_ece='ece',
                                    draw_cls_metrics=True,
                                    draw_bin_importance='none',
                                    plot_gap=False,
                                    plot_confidence_interval=True,
                                    emphasize_identity=False,
                                    calibration_type='confidence',
                                    metric_fontsize=22,
                                    ):
    """Draws a reliability diagram into a subplot."""
    if calibration_type == 'confidence':
        xlabel = '$p$'
        ylabel = '$P(Y=\hat{Y} \mid \max (\hat{P}(X)) = p)$'
    elif isinstance(calibration_type, int):
        k = calibration_type
        xlabel = r"$s$"
        ylabel = r"$\hat{r}(s)$"
        xlabel = ""
        ylabel = ""
    else:
        raise ValueError(f'calibration_type={calibration_type} not implemented.')
        
    accuracies = bin_data["accuracies"]
    confidences = bin_data["confidences"]
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    widths = bin_size
    alphas = 0.3
    min_count = np.min(counts)
    max_count = np.max(counts)
    normalized_counts = (counts - min_count) / (max_count - min_count + 1e-8)

    if draw_bin_importance == "alpha":
        alphas = 0.2 + 0.8*normalized_counts
    elif draw_bin_importance == "alpha_exclude_endbin":
        counts = counts.copy()
        counts[0] = counts[1]
        counts[-1] = counts[-2]
        min_count = np.min(counts)
        max_count = np.max(counts)
        normalized_counts = (counts - min_count) / (max_count - min_count + 1e-8)
        alphas = 0.2 + 0.8*normalized_counts
    elif draw_bin_importance == "width":
        widths = 0.1*bin_size + 0.9*bin_size*normalized_counts
    elif draw_bin_importance == 'counts':
        alphas = normalized_counts
    elif draw_bin_importance == 'none':
        alphas = 0

    colors_black = np.zeros((len(counts), 4))
    colors_black[:, [0,1,2]] = tuple(x/255 for x in (100,100,100))
    colors_black[:, 3] = alphas

    colors_binsize = np.zeros((len(counts), 4))
    colors_binsize[:, [0,1,2]] =tuple(x/255 for x in (100,100,100))
    colors_binsize[:, 3] = alphas

    color_calibration_curve = (255/255, 0/255, 0/255) # red

    ax.plot([0,1], [0,1], linestyle="--", linewidth=2, color="gray", zorder=1)

    if plot_gap:
        ax.bar(positions, np.abs(accuracies - confidences), bottom=np.minimum(accuracies, confidences), width=widths, edgecolor=colors_black, color=colors_black, linewidth=1, label="Gap")

        ax.bar(positions, 0, bottom=accuracies, width=widths, edgecolor=colors_binsize, color=colors_binsize, alpha=1.0, linewidth=3, label="Accuracy")
    else:
        # show pseudo-count size as shades under the curve. darker -> larger counts
        ax.bar(positions, accuracies, width=widths, edgecolor=colors_binsize, color=colors_binsize, linewidth=0) # , edgecolor=colors_red, linewidth=0
        # # show pseudo-count size as another curve. does not look good.
        # ax2 = ax.twinx()
        # ax2.plot(positions, np.log(normalized_counts), color='gray')

        ## draw confidence interval
        if plot_confidence_interval:
            assert('accuracies_low' in bin_data and 'accuracies_high' in bin_data)
            low, high = bin_data['accuracies_low'], bin_data['accuracies_high']
            ax.fill_between(positions, low, high, color='b', alpha=0.2)

        bins = bins.copy()
        bins[0] -= bin_size
        bins[-1] += bin_size
        ax.stairs(accuracies, edges=bins, color=color_calibration_curve, linewidth=4, fill=False, baseline=False)
        if emphasize_identity:
            ax.scatter(confidences, confidences, color='k', marker='o', s=80)

    # ax.set_aspect("equal")
    ax.set_aspect(aspect='auto', adjustable='box') 
    
    if draw_ece:
        ece = bin_data["ece"]
        ece_exclude_endbin = compute_ece_from_bin_data(bin_data, exclude_endbin=True)
        if draw_ece == 'ece_exclude_endbin':
            s = r"ECE$\!^*\!$: " + f"{ece_exclude_endbin:.3f}"
        elif draw_ece == 'ece':
            s = r"ECE: " f"{ece:.3f}"
        elif draw_ece == 'both':
            s = r"ECE, ECE$\!^*\!$: " + f"{ece:.3f}, {ece_exclude_endbin:.3f}"
        else:
            raise ValueError(f"Invalid {draw_ece}")
        ax.text(0.04, .97, s, color="black", ha="left", va="top", transform=ax.transAxes, fontsize=metric_fontsize)

    if draw_cls_metrics:
        acc = bin_data['accuracy']
        brier = bin_data['brier_score']
        s = r"BS: " + f"{brier:.3f}"
        ax.text(0.12, 0.87, s, color="black", ha="left", va="top", transform=ax.transAxes, fontsize=metric_fontsize)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ylabel = r"$E\left[ Y \mid g(X) \right]$"
    ylabel = r'$\hat{r}(s)$'
    yticks = [.25, .5, .75, 1]
    ytick_labels = [.25, .5, .75, ylabel]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels) 
    for label in ax.get_yticklabels():
        if label.get_text() == ylabel:
            label.set_rotation(90)
            label.set_fontsize(label.get_fontsize() * 1.5)
            label.set_va('center')

    # Set custom x-ticks and labels
    xlabel = r'$g(X)$'
    xlabel = r'$s$'
    xticks = [0.25, 0.5, 0.75, 1]
    xtick_labels = [0.25, 0.5, 0.75, xlabel]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels) 
    for label in ax.get_xticklabels():
        if label.get_text() == xlabel:
            label.set_fontsize(label.get_fontsize() * 1.5)

    

def plt_reliability_diagram_for_pathologies(
        pathologies,
        calibrate_over_dist=True,
        calibration_type=1,
        num_bins=10,
        emphasize_identity=True,
        bootstrap_num_samples=1,
        bootstrap_alpha=0.05,
        draw_cls_metrics=True,
        draw_ece='ece',
        nrows='auto',
        **kwargs
    ):
    """Plot reliability diagram for each pathology in `pathologies`.

    if `calibrate_over_dist` is true, assumes `u_dist` is supplied in `kwargs`
    if `calibrate_over_dist` is false, assumes `y, u_prob` are supplied in `kwargs`
        
    `calibration_type`: 0,1 for class-wise ECE, 'confidence' for confidence ECE.
    """

    plt_params = {
        'font.size': 25,
        'font.family': 'Times New Roman',
        "legend.frameon": False,
        "axes.titlepad": 10,
        # bar plot
        "patch.force_edgecolor": False,
        'patch.linewidth': 2,  # Edge line width for bar patches
        # tick label size
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
    }

    with plt.rc_context(plt_params):
        if nrows == 'auto':
            nrows = 1 if len(pathologies)==1 else 2
        ncols = len(pathologies)//nrows
        fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, .5+4*nrows), sharex=True, sharey=True)
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        for axi, pathology in enumerate(pathologies):
            ax = axs.reshape(-1)[axi]
            bin_data = compute_calibration_with_bootstrap(
                pathology=pathology,
                calibrate_over_dist=calibrate_over_dist,
                calibration_type=calibration_type,
                num_bins=num_bins,
                bootstrap_num_samples=bootstrap_num_samples,
                bootstrap_alpha=bootstrap_alpha,
                **kwargs,
            )
            plt_reliability_diagram_subplot(ax, bin_data, draw_bin_importance='alpha_exclude_endbin', calibration_type=calibration_type, emphasize_identity=emphasize_identity, draw_cls_metrics=draw_cls_metrics, draw_ece=draw_ece)

            pathology = pathology.split(' ')[-1].capitalize()
            ax.set_title(f"{pathology}", fontsize=40)
            ax.grid()

        for i in range(nrows):
            for j in range(ncols):
                ax = axs.reshape(nrows,ncols)[i,j]
                if i != nrows-1:
                    ax.set_xlabel('')
                if j != 0:
                    ax.set_ylabel('')

        fig.tight_layout(pad=.5)

    return fig, axs
        


def plt_calibration_map_between_dcps(
        ax,
        P,
        source_text,
        target_text,
        vertical_gap=.3,
        min_width=2,
        max_width=15,
        fontsize=20,
        bbox_pad=.5,
        display_line_threshold=1e-8,
    ):

    M = len(source_text)
    N = len(target_text)

    P_max = P.max()
    P_marginal_source = P.sum(1)
    P_marginal_target = P.sum(0)

    xs = np.stack(( np.zeros(M), np.arange(M) * vertical_gap )).T
    xt = np.stack(( np.ones(M)*3,  np.arange(N) * vertical_gap )).T

    # Generate colors for source points
    colors = plt.cm.rainbow(np.linspace(0, 1, xs.shape[0]))

    # cmap = plt.get_cmap('tab10'); colors = [cmap(i) for i in range(N)]
    cmap = plt.get_cmap('coolwarm'); colors = [cmap(i/N) for i in range(N)]


    for i in range(M):
        for j in range(N):
            if P[i, j] / P_max > display_line_threshold:
                width = (P[i, j] / P_max) * (max_width-min_width) + min_width
                ax.plot([xs[i, 0],
                        xt[j, 0]], 
                        [xs[i, 1] - vertical_gap/3/2 + j*vertical_gap/3/N, 
                        xt[j, 1] - vertical_gap/3/2 + i*vertical_gap/3/M], 
                        linewidth=width, color=colors[i], alpha=1)


    # Function to create text with custom bounding box
    def add_text_with_bbox(x, y, text, weight, ha, edgecolor, fontsize, weight_fontsize):
        bbox_props = dict(facecolor='white', alpha=1, edgecolor=edgecolor, linewidth=min_width, pad=bbox_pad, boxstyle=f'round,pad={bbox_pad}')
        bbox = ax.text(x, y, f"""“{text}”""", ha=ha, va='center', fontsize=fontsize, bbox=bbox_props)
        # Add weight text below the bounding box
        ax.text(x, y-0.05, f"{weight:.3f}", ha=ha, va='top', alpha=1, fontsize=weight_fontsize, color='k')

    # Plot source text boxes with marginal weights
    for i, (xy, text, weight) in enumerate(zip(xs, source_text, P_marginal_source)):
        add_text_with_bbox(xy[0]-0.03, xy[1], text, weight, 'right', colors[i], fontsize, fontsize/2)

    for i, (xy, text, weight) in enumerate(zip(xt, target_text, P_marginal_target)):
        add_text_with_bbox(xy[0]-0.03, xy[1], text, weight, 'left', 'k', fontsize, fontsize/2)

    xext = 3.5
    yext = .5 * vertical_gap
    ax.set_xlim(0-xext, 3+xext)
    ax.set_ylim(min(xs[:, 1].min(), xt[:, 1].min()) - yext, 
                max(xs[:, 1].max(), xt[:, 1].max()) + yext)
    ax.axis('off')



def calibrate_ot_prepare(
        pathology,
        u_dist,
        u_dcp,
        source_dist,
        target_dist,
        target_weight,
        ## supplied to `compute_calibration_with_bootstrap` all arguments except `u_dist` and `pathology`
        compute_calibration_with_bootstrap_kwargs={},
    ):
    """Computes the cost function C between source and target distributions,
        and prepares the source and target weights a,b for optimal transport.

        `u_dist`    {(pathology, 'pred'/'true): [Beta(), ...]}
            a dictionary of dataset of distributions
        `source_dist` `target_dist` Dict[str, torch.distributions.Distribution]
            dictionary of dcp and corresponding distributions
    """

    if 'include_cls_metrics' in compute_calibration_with_bootstrap_kwargs:
        compute_calibration_with_bootstrap_kwargs = compute_calibration_with_bootstrap_kwargs.copy()
        compute_calibration_with_bootstrap_kwargs['include_cls_metrics'] = False

    u_dist_original = {k: v.copy() for k, v in u_dist.items()}

    K = len(source_dist)
    L = len(target_dist)

    ## construct cost function
    cost_bin_data = {}
    C = np.zeros((K, L))

    source_dist_items = list(source_dist.items())
    target_dist_items = list(target_dist.items())
    loop_iter = itertools.product(
        list(range(len(source_dist_items))),
        list(range(len(target_dist_items))),
    )
    loop_iter = list(loop_iter)
    loop_iter = tqdm(loop_iter, total=len(loop_iter), desc="Compute Cost Function")

    for ind_k, ind_l in loop_iter:
        dcp_k, u_k = source_dist_items[ind_k]
        dcp_l, v_l = target_dist_items[ind_l]
        u_dist = {k: v.copy() for k, v in u_dist_original.items()}
        for k in list(u_dist.keys()):
            if k[1] == 'pred': # only substitute u_k->v_l for prediction
                u_dist[k] = [v_l if dcp==dcp_k else u for dcp, u in zip(u_dcp[k], u_dist[k])]
        bin_data = compute_calibration_with_bootstrap(
            u_dist=u_dist,
            pathology=pathology,
            **compute_calibration_with_bootstrap_kwargs,
        ) 
        cost_bin_data[(ind_k, ind_l)] = bin_data
        C[ind_k, ind_l] = bin_data['ece']

    # normalize row of cost function by source weight
    counts = dict(Counter(u_dcp[(pathology, 'pred')]).most_common())
    counts = {k: counts[k] if k in counts else 0 for k in source_dist.keys()}
    counts = np.array(list(counts.values()))
    # handles division by 0 -> set to divide by 1
    C = (C - C[0,0]) / np.where(counts.reshape(-1, 1) == 0, 1, counts.reshape(-1, 1)) * counts.sum()

    ## create source/target weights / probability vectors. 
    a = counts
    a = a / a.sum()

    if isinstance(target_weight, np.ndarray):
        b = target_weight.copy()
    elif target_weight == 'equal_to_a':
        b = a.copy()
    else:
        raise ValueError(f'Invalid {target_weight}')

    outputs = {
        'C': C,
        'a': a,
        'b': b,
    }

    return outputs


def calibrate_ot_solve(
        ot_type,
        a,
        b,
        C,
        verbose=False,
    ):
    """Solve the optimal transport problem with the given parameters.
    `ot_type` is a string that specifies the type of optimal transport to use.
        - `emd`: Earth Mover's Distance
        - `unbsink`: Unbalanced Sinkhorn with default parameters
        - `unbsinkstab`: Unbalanced Sinkhorn with stabilized parameters
        - `unbsinkscaling`: Unbalanced Sinkhorn with scaling regularization
    The string can also contain additional parameters in the format:
        `fn=emd|unbsink|unbsinkstab|unbsinkscaling_eps=0.1_gamma=0.1_div=kl`

    `a` and `b` are the source and target weights, respectively.
    `C` is the cost matrix between the source and target distributions.
    Returns a dictionary with the transport matrix `P` and the loss value.
    """

    kvs = parse_kv_from_string(ot_type)
    ot_fn_type = kvs['fn']
    if ot_fn_type == 'emd':
        ot_fn = ot.emd
        ot_fn_kwargs = {}
    elif ot_fn_type.startswith('unbsink'):
        ot_fn = ot.sinkhorn_unbalanced
        if ot_fn_type == 'unbsink':
            method = 'sinkhorn'
        elif ot_fn_type == 'unbsinkstab':
            method = 'sinkhorn_stabilized'
        elif ot_fn_type == 'unbsinkscaling':
            method = 'sinkhorn_reg_scaling'
        else:
            raise ValueError(f'Invalid ot_fn_type: {ot_fn_type}.')
        ot_fn_kwargs = {
            'reg': float(kvs['eps']),
            'reg_m': (float('inf'), float(kvs['gamma'])),
            'method': method,
            'div': kvs.get('div', 'kl'), # kl or l2
        }
    else:
        raise ValueError(f'Invalid ot_fn_type: {ot_fn_type}.')


    start = time.time()
    if verbose:
        print(f"Running {ot_fn} with kwargs:\m{ot_fn_kwargs}")
    P = ot_fn(a, b, C, **ot_fn_kwargs)
    if verbose:
        print(f'OT finish in {time.time() - start:.3f} seconds')
    loss = (P*C).sum()
    if verbose:
        print('transport matrix:')
        print(P.round(3))
        print('mean(|t*1-a|):   ', np.mean(np.abs(P.sum(1)-a)))
        print('mean(|t^T*1-b|): ', np.mean(np.abs(P.sum(0)-b)))

    return {'P': P,
            'loss': loss}


def transport_certainty_phrases(
        P,
        dcp_list,
        source_dist,
        target_dist,
    ):
    """Transport a list of confidences ["maybe", "likely", ...] according to 
        the transport plan `P`. """
    K, L = len(source_dist), len(target_dist)

    np.random.seed(0)
    dcp_transported_list = ['']*len(dcp_list)
    for ind_k, (dcp_k, u_k) in enumerate(source_dist.items()):
        mask_is_uk = [x == dcp_k for x in dcp_list]
        num_uk = sum(mask_is_uk)
        if num_uk < 1:
            continue
        prob = P[ind_k, :]
        if (prob==0).all(): # in case prob all zeros, set to uniform, will give error during sampling
            print(f'invalid prob: {prob}. set to uniform.')
            prob = np.ones(L)
        prob = prob / prob.sum()
        target_dcp_sampled = np.random.choice(list(target_dist.keys()), size=num_uk, p=prob)
        target_dist_sampled_iter = iter(target_dcp_sampled)
        
        for i, (d, m) in enumerate(zip(dcp_transported_list, mask_is_uk)):
            if m:
                if dcp_transported_list[i] != '':
                    raise ValueError(f'Overwriting transported dcp list at index {i}')
                dcp_transported_list[i] = next(target_dist_sampled_iter)
    return dcp_transported_list

def get_reliability_diagram_subplot_kwargs(calibrate_over):
    """ reliability diagram kwargs for `llm` related runs"""
    if calibrate_over == 'dist':
        return {
            'draw_bin_importance': 'alpha',
            'calibration_type': 1,
            'emphasize_identity': False,
            'plot_confidence_interval': True,
            'draw_cls_metrics': False,
            'draw_ece': 'ece',
            'metric_fontsize': 20,
        }
    elif calibrate_over == 'prob':
        return {
            'draw_bin_importance': 'alpha',
            'calibration_type': 1,
            'emphasize_identity': False,
            'plot_confidence_interval': False,
            'draw_cls_metrics': False,
            'draw_ece': 'ece',
            'metric_fontsize': 20,
        }
    else:
        raise ValueError(f'Invalid calibrate_over: {calibrate_over}')


def plt_ot_results(r, answers_subdir_name):
    """Given a TaskResult `r`, output figure of the calibration process to `answers_subdir_name`"""

    assert('test' in answers_subdir_name)

    rs = r.get_answers_list()
    r_test = rs['test']
    r_ot = rs[answers_subdir_name]

    bin_data_original = compute_calibration_with_bootstrap(
        u_dist=r_test.classification_data['u_dist'],
        pathology=r_test.model_name,
        **r_test.calibration_bin_data_dist_kwargs,
    )
    bin_data_transported = compute_calibration_with_bootstrap(
        u_dist=r_ot.classification_data['u_dist'],
        pathology=r_ot.model_name,
        **r_ot.calibration_bin_data_dist_kwargs,
    )

    fig, axs = plt_calibration_results(
        bin_data_original=bin_data_original,
        bin_data_transported=bin_data_transported,
        ot_outputs=r_ot.ot_outputs,
        source_dist=r.certainty_phrase_distributions,
        target_dist=r.certainty_phrase_distributions,
        plt_reliability_diagram_subplot_kwargs=get_reliability_diagram_subplot_kwargs('dist'),
        plt_params_version='llm',
    )

    fig_path = os.path.join(r.save_dir, answers_subdir_name, 'ot_results.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')



def calibrate_ot(
        pathology,
        u_dist,
        u_dcp,
        source_dist,
        target_dist,
        target_weight,
        ## supplied to `compute_calibration_with_bootstrap` all arguments except `u_dist` and `pathology`
        compute_calibration_with_bootstrap_kwargs={},
    ):
    """Legacy code work for `rad` only for now.
        `u_dist`    {(pathology, 'pred'/'true): [Beta(), ...]}
            a dictionary of dataset of distributions
        `source_dist` `target_dist` Dict[str, torch.distributions.Distribution]
            dictionary of dcp and corresponding distributions
    """

    if 'include_cls_metrics' in compute_calibration_with_bootstrap_kwargs:
        compute_calibration_with_bootstrap_kwargs = compute_calibration_with_bootstrap_kwargs.copy()
        compute_calibration_with_bootstrap_kwargs['include_cls_metrics'] = False

    u_dist_original = {k: v.copy() for k, v in u_dist.items()}

    K = len(source_dist)
    L = len(target_dist)

    ## construct cost function
    cost_bin_data = {}
    C = np.zeros((K, L))

    source_dist_items = list(source_dist.items())
    target_dist_items = list(target_dist.items())
    loop_iter = itertools.product(
        list(range(len(source_dist_items))),
        list(range(len(target_dist_items))),
    )
    loop_iter = list(loop_iter)
    loop_iter = tqdm(loop_iter, total=len(loop_iter), desc="Compute Cost Function")

    for ind_k, ind_l in loop_iter:
        dcp_k, u_k = source_dist_items[ind_k]
        dcp_l, v_l = target_dist_items[ind_l]
        u_dist = {k: v.copy() for k, v in u_dist_original.items()}
        for k in list(u_dist.keys()):
            if k[1] == 'pred': # only substitute u_k->v_l for prediction
                u_dist[k] = [v_l if x==u_k else x for x in u_dist[k]]
        bin_data = compute_calibration_with_bootstrap(
            u_dist=u_dist,
            pathology=pathology,
            **compute_calibration_with_bootstrap_kwargs,
        ) 
        cost_bin_data[(ind_k, ind_l)] = bin_data
        C[ind_k, ind_l] = bin_data['ece']

    # normalize row of cost function by source weight
    counts = dict(Counter(u_dist[(pathology, 'pred')]).most_common())
    counts = {k: counts[k] if k in counts else 0 for k in source_dist.values()}
    counts = np.array(list(counts.values()))
    C = (C - C[0,0]) / counts.reshape(-1, 1) * counts.sum()

    ## create source/target weights / probability vectors. 
    a = counts
    a = a / a.sum()

    if isinstance(target_weight, np.ndarray):
        b = target_weight.copy()
    elif target_weight == 'equal_to_a':
        b = a.copy()
    else:
        raise ValueError(f'Invalid {target_weight}')
    

    ## solve OT problem
    start = time.time()
    P = ot.emd(a, b, C)
    # P = ot.sinkhorn(a, b, C, 1e-2, method='sinkhorn_epsilon_scaling')
    print(f'OT finish in {time.time() - start:.3f} seconds')
    loss = (P*C).sum()

    print('transport matrix:')
    print(P.round(3))
    print('mean(|t*1-a|):   ', np.mean(np.abs(P.sum(1)-a)))
    print('mean(|t^T*1-b|): ', np.mean(np.abs(P.sum(0)-b)))


    ## transport `u_dist` according to the transport plan `P`
    np.random.seed(0)
    u_dist_transported = {k: v.copy() for k,v in u_dist.items()}
    for ds_slice in list(u_dist.keys()):
        if ds_slice[1] == 'true': # just alter `pred` 
            continue
        for ind_k, (dcp_k, u_k) in enumerate(source_dist.items()):
            mask_is_uk = [x == dcp_k for x in u_dcp[ds_slice]]
            num_uk = sum(mask_is_uk)
            prob = P[ind_k, :]
            if (prob==0).all(): # in case prob all zeros, set to uniform, will give error during sampling
                print(f'invalid prob: {prob}. set to uniform.')
                prob = np.ones(L)
            prob = prob / prob.sum()
            # print(ind_k, prob, sum(mask_is_uk))
            target_dcp_sampled = np.random.choice(list(target_dist.keys()), size=num_uk, p=prob)
            target_dist_sampled = [target_dist[k] for k in target_dcp_sampled]
            target_dist_sampled_iter = iter(target_dist_sampled)
            u_dist_transported[ds_slice] = [next(target_dist_sampled_iter) if is_uk else x 
                                            for x, is_uk in zip(u_dist_transported[ds_slice], mask_is_uk )]

    outputs = {
        'C': C,
        'a': a,
        'b': b,
        'P': P,
        'loss': loss,
        'source_dist': source_dist,
        'target_dist': target_dist,
        'u_dist_transported': u_dist_transported,
        'cost_bin_data': cost_bin_data,
    }

    return outputs



def plt_calibration_results(
        bin_data_original,
        bin_data_transported,
        ot_outputs,
        source_dist,
        target_dist,
        fig_ylabel='',
        plt_reliability_diagram_subplot_kwargs=None,
        plt_params_version='rad',
    ):
    """Plot reliability curve before/after calibration.
        calibration cost matrix and calibration map."""


    K = len(source_dist)
    L = len(target_dist)

    if plt_params_version == 'rad':
        plt_params = {
            'font.size': 25,
            'font.family': 'Times New Roman',
            "legend.frameon": False,
            "axes.titlepad": 10,
            # bar plot
            "patch.force_edgecolor": False,
            'patch.linewidth': 2,  # Edge line width for bar patches
            # tick label size
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
        }
        cost_cell_fontsize = 14
        calibration_map_fontsize = 20
    elif plt_params_version == 'llm':
        plt_params = {
            'font.size': 25,
            'font.family': 'Times New Roman',
            "legend.frameon": False,
            "axes.titlepad": 10,
            # bar plot
            "patch.force_edgecolor": False,
            'patch.linewidth': 2,  # Edge line width for bar patches
            # tick label size
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
        }
        cost_cell_fontsize = 8
        calibration_map_fontsize = 9
    else:
        raise ValueError(f"Invalid plt_params_version: {plt_params_version}")

    if plt_reliability_diagram_subplot_kwargs is None:
        plt_reliability_diagram_subplot_kwargs = {
            'draw_bin_importance': 'alpha_exclude_endbin',
            'calibration_type': 1,
            'emphasize_identity': False,
            'draw_cls_metrics': True,
            'draw_ece': 'ece_exclude_endbin',
            'metric_fontsize': 20,
        }

    with plt.rc_context(plt_params):
                
        nrows = 1
        ncols = 4
        w = 4.5
        fig, axs = plt.subplots(nrows, ncols, figsize=(.5 + ncols*w, nrows*w))
        
        ## reliability diagram for original data
        ax = axs[0]
        plt_reliability_diagram_subplot(ax, bin_data_original, **plt_reliability_diagram_subplot_kwargs)
        ax.grid()
        ax.set_title('Before')
        
        ax = axs[1]
        C = ot_outputs['C']
        plt_kernel_matrix_one(fig, ax, C, n_ticks=5, custom_ticks=False, annotate=False, cmap='bwr')
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                ax.annotate(f'{C[i,j]:.2f}', xy=(j, i),
                            fontsize=cost_cell_fontsize,
                            horizontalalignment='center',
                            verticalalignment='center')
        ax.set_title('Cost')
        ax.set_yticks(np.arange(K), [f"“{x}”" for x in list(source_dist.keys())])
        ax.set_xticks(np.arange(L), [f"“{x}”" for x in list(target_dist.keys())], ha='left', rotation=-30) 
        
        ax = axs[2]
        P = ot_outputs['P']
        plt_calibration_map_between_dcps(
                ax,
                P,
                source_text=list(source_dist.keys()),
                target_text=list(target_dist.keys()),
                vertical_gap=.3,
                min_width=1,
                max_width=15,
                fontsize=calibration_map_fontsize,
                bbox_pad=.6,
                display_line_threshold=1e-3,
            )
        ax.set_title('Calibration Map')
        # ax.set_title(f'Loss = {ot_outputs["loss"]:.4f}')
        
        ax = axs[3]
        plt_reliability_diagram_subplot(ax, bin_data_transported, **plt_reliability_diagram_subplot_kwargs)
        ax.grid()
        ax.set_title('After')

        if fig_ylabel:
            fig.text(-.02, 0.5, fig_ylabel, va='center', rotation='vertical', fontsize=35)
        fig.tight_layout(pad=0.5)

    return fig, axs



def calibrate_ot_on_task_result(
        r,
        ot_type_list,
        refresh=False,
    ):
    """Given TaskResult `r`, calibrate according to `ot_type`"""

    if not isinstance(ot_type_list, list):
        ot_type_list = [ot_type_list]

    if not refresh:
        print(f'len(ot_type_list) (before): {len(ot_type_list)}')
        ot_type_list = [
            ot_type for ot_type in ot_type_list
            if not os.path.isdir(os.path.join(r.save_dir, f'test_{ot_type}'))
        ]
        print(f'len(ot_type_list) (after): {len(ot_type_list)}')
        if len(ot_type_list) == 0:
            print(f'ot_type_list already done: {ot_type_list}')
            return

    r.train_test_split(.5)
    rs = r.get_answers_list()
    r_train = rs['train']
    r_test = rs['test']

    source_dist = r.certainty_phrase_distributions
    target_dist = r.certainty_phrase_distributions
    K, L = len(source_dist), len(target_dist)

    ot_outputs_shared = calibrate_ot_prepare(
        pathology=r.model_name,
        u_dist=r_train.classification_data['u_dist'],
        u_dcp=r_train.classification_data['u'],
        source_dist=source_dist,
        target_dist=target_dist,
        target_weight='equal_to_a',
        compute_calibration_with_bootstrap_kwargs=r.calibration_bin_data_dist_kwargs,
    )

    for ot_type in tqdm(ot_type_list, total=len(ot_type_list), desc="Solve OT (different ot_type)"):
        transported_save_subdir = f'test_{ot_type}'
        ot_outputs = {k: v.copy() for k, v in ot_outputs_shared.items()}

        try: # can possibly fail due to incorrect combination of parameters.
            ot_solve_outputs = calibrate_ot_solve(
                ot_type=ot_type,
                a=ot_outputs['a'],
                b=ot_outputs['b'],
                C=ot_outputs['C'],
                verbose=False,
            )
            if np.isnan(ot_solve_outputs['P']).any():
                continue
        except:
            continue
        ot_outputs.update(ot_solve_outputs)

        ## apply transport map to certainty phrases
        dcp_transported_list = transport_certainty_phrases(
            P=ot_outputs['P'],
            dcp_list=r_test.classification_data['u'][(r.model_name, 'pred')],
            source_dist=source_dist,
            target_dist=target_dist,
        )

        # modify answers with transported answers
        transported_answers = []
        for d, dcp in zip(r_test.answers, dcp_transported_list):
            d = d.copy()
            d['confidence'] = dcp
            transported_answers.append(d)
        if len(r_test.answers) != len(dcp_transported_list):
            raise ValueError(f'before/after transport should have same length of answers.')

        # save results to `<save_dir>/<test_{ot_type}>/`
        r.save_answers(
            transported_save_subdir,
            transported_answers,
        )
        
        # save ot_outputs to  `<save_dir>/<test_{ot_type}>/ot_outputs.pkl`
        with open(os.path.join(r.save_dir, transported_save_subdir, 'ot_outputs.pkl'), 'wb') as f:
            pickle.dump(ot_outputs, f, protocol=pickle.HIGHEST_PROTOCOL)

        # save ot results figure to  `<save_dir>/<test_{ot_type}>/ot_results.png`
        plt_ot_results(r, transported_save_subdir)




def calibrate_probs_on_verbprob(
        r,
        calibration_type,
        refresh=False,
    ):
    """Apply platt scaling to verbalizing probabilities """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    if calibration_type not in ['platt']:
        raise ValueError(f'Invalid calibration_type: {calibration_type}')

    calibrated_save_subdir = f'test_{calibration_type}'

    if (not refresh) and os.path.isdir(os.path.join(r.save_dir, calibrated_save_subdir)):
        print(f'Already calibrated {calibrated_save_subdir}')
        return

    ## split data
    r.train_test_split(.5)
    rs = r.get_answers_list()
    r_train = rs['train']
    r_test = rs['test']

    def get_probs_and_labels(run):
        probs = run.classification_data['u_prob'][(run.model_name, 'pred')]
        ys = run.classification_data['y'][(run.model_name, 'true')]
        return probs, ys

    probs_train, ys_train = get_probs_and_labels(r_train)
    probs_test, ys_test = get_probs_and_labels(r_test)
    probs_train = probs_train.reshape(-1, 1)
    probs_test = probs_test.reshape(-1, 1)

    ## train calibrator
    platt_scaler = LogisticRegression(solver='lbfgs')
    platt_scaler.fit(probs_train, ys_train)
    calibrated_probs_test = platt_scaler.predict_proba(probs_test)[:, 1]

    brier_score_before = brier_score_loss(ys_test, probs_test)
    print(f'Brier score before calibration: {brier_score_before:.3f}')
    brier_score_after = brier_score_loss(ys_test, calibrated_probs_test)
    print(f'Brier score after calibration: {brier_score_after:.3f}')

    ## calibrate certainty phrases (e.g., probabilities)
    confidences = r_test.classification_data['u'][(r_test.model_name, 'pred')]
    confidences = np.array([float(x) for x in confidences]).reshape(-1, 1)
    confidences_calibrated = platt_scaler.predict_proba(confidences)[:, 1]
    confidences_calibrated = [str(x) for x in confidences_calibrated.round(3).tolist()]

    # modify answers with transported answers
    calibrated_answers = []
    for d, confidence in zip(r_test.answers, confidences_calibrated):
        d = d.copy()
        d['confidence'] = confidence
        calibrated_answers.append(d)
    if len(r_test.answers) != len(calibrated_answers):
        raise ValueError(f'before/after calibration should have same length of answers.')

    # save results to `<save_dir>/<test_{calibration_type}>/`
    r.save_answers(
        calibrated_save_subdir,
        calibrated_answers,
    )

    ## some plotting before/after calibration.
    r_test = r.get_answers_list()['test']
    r_calib = r.get_answers_list()['test_platt']
    calibrate_over = 'prob'
    fig, axs = plt.subplots(1, 2, figsize=(2*4, 4))
    for axi, rt in enumerate([r_test, r_calib]):
        ax = axs[axi]
        if calibrate_over == 'prob':
            bin_data = rt.calibration_bin_data_prob
        elif calibrate_over == 'dist':
            bin_data = rt.calibration_bin_data_dist
        plt_reliability_diagram_subplot(ax, bin_data, **get_reliability_diagram_subplot_kwargs('prob'))
        ax.grid()
    fig_path = os.path.join(r.save_dir, calibrated_save_subdir, 'calibrate_results.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')




def calibrate_probs_on_verbconf(
        r,
        calibration_type,
        refresh=False,
    ):
    """Apply platt scaling/histogram binning to verbalizing phrases """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    if calibration_type not in ['platt', 'histbinning']:
        raise ValueError(f'Invalid calibration_type: {calibration_type}')

    calibrated_save_subdir = f'test_{calibration_type}'

    if (not refresh) and os.path.isdir(os.path.join(r.save_dir, calibrated_save_subdir)):
        print(f'Already calibrated {calibrated_save_subdir}')
        return

    ## split data
    r.train_test_split(.5)
    rs = r.get_answers_list()
    r_train = rs['train']
    r_test = rs['test']

    def get_probs_and_labels(run):
        probs = run.classification_data['u_prob'][(run.model_name, 'pred')]
        ys = run.classification_data['y'][(run.model_name, 'true')]
        return probs, ys

    probs_train, ys_train = get_probs_and_labels(r_train)
    probs_test, ys_test = get_probs_and_labels(r_test)
    probs_train = probs_train.reshape(-1, 1)
    probs_test = probs_test.reshape(-1, 1)

    ## train & apply calibrator
    if calibration_type == 'platt':
        platt_scaler = LogisticRegression(solver='lbfgs')
        platt_scaler.fit(probs_train, ys_train)
        calibrated_probs_test = platt_scaler.predict_proba(probs_test)[:, 1]
    elif calibration_type == 'histbinning':
        # {phrase: calibrated_prob}
        histogram_binning_calibrator = {}
        for dcp, u_dist in r_train.certainty_phrase_distributions.items():
            u_dist_mean = u_dist.mean.item()
            ys_train_predu = ys_train[(probs_train == u_dist_mean).squeeze()]
            if ys_train_predu.shape[0] > 0:
                calibrated_prob = ys_train_predu.mean()
            else:
                calibrated_prob = u_dist_mean
            histogram_binning_calibrator[u_dist_mean] = calibrated_prob 
        calibrated_probs_test = np.array(
            [histogram_binning_calibrator[p] for p in probs_test.squeeze().tolist()])

    brier_score_before = brier_score_loss(ys_test, probs_test)
    print(f'Brier score before calibration: {brier_score_before:.3f}')
    brier_score_after = brier_score_loss(ys_test, calibrated_probs_test)
    print(f'Brier score after calibration: {brier_score_after:.3f}')

    ## calibrate certainty phrases (e.g., probabilities)
    confidences_calibrated = [str(x) for x in calibrated_probs_test.round(3).tolist()]

    # modify answers with transported answers
    calibrated_answers = []
    for d, confidence in zip(r_test.answers, confidences_calibrated):
        d = d.copy()
        d['confidence'] = confidence
        calibrated_answers.append(d)
    if len(r_test.answers) != len(calibrated_answers):
        raise ValueError(f'before/after calibration should have same length of answers.')

    # save results to `<save_dir>/<test_{calibration_type}>/`
    r.save_answers(
        calibrated_save_subdir,
        calibrated_answers,
        additional_config={'prompt_type': 'verbprobtop1', 'certainty_phrase_source': 'unspecified_delta'},
    )

    ## some plotting before/after calibration.
    r_test = r.get_answers_list()['test']
    r_calib = r.get_answers_list()['test_platt']
    calibrate_over = 'prob'
    fig, axs = plt.subplots(1, 2, figsize=(2*4, 4))
    for axi, rt in enumerate([r_test, r_calib]):
        ax = axs[axi]
        if calibrate_over == 'prob':
            bin_data = rt.calibration_bin_data_prob
        elif calibrate_over == 'dist':
            bin_data = rt.calibration_bin_data_dist
        plt_reliability_diagram_subplot(ax, bin_data, **get_reliability_diagram_subplot_kwargs('prob'))
        ax.grid()
    fig_path = os.path.join(r.save_dir, calibrated_save_subdir, 'calibrate_results.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
