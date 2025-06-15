from typing import Callable, List
from functools import cached_property
import os
import shutil
import re
import pickle
import json
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch

from .utils import parse_kv_from_string, dict_iterated_getitem
from .calibration import compute_calibration_with_bootstrap, get_dcp_dist




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



def train_test_split_pathology_labels_data(
        info,
        test_size,
        verbose=False,
    ):
    """Convert `info` that contains certainty predictions & labels to train/test splits.
        Specifically, use stratified sampling wrt confidence distributions for each pathology.
    """
    from sklearn.model_selection import train_test_split
    from collections import Counter

    info_train = {}
    info_test = {}

    def index_iterable_with_ndarray(L, inds):
        if isinstance(L, (tuple, list)):
            Lp = [L[i] for i in inds]
        elif isinstance(L, np.ndarray):
            Lp = L[inds]
        return Lp

    inds_train = {}
    inds_test = {}

    for pathology in list(set([x[0] for x in info['u'].keys()])):

        dcp_data = info['u'][(pathology, 'pred')]
        N = len(dcp_data)

        ## in case there is DCP that occurs <=1 times, stratified sampling gives error
        # ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
        # convert this very infrequent DCP to the most common DCP for the purpose of stratified sampling.
        stratify = dcp_data
        counts = Counter(stratify).most_common()
        stratify_count_is_one = set([k for k, v in dict(counts).items() if v <= 1])
        stratify = [counts[0][0] if x in stratify_count_is_one else x for x in stratify]
        train_inds, test_inds = train_test_split(np.arange(N), test_size=test_size, shuffle=True, random_state=0, stratify=stratify)

        inds_train[pathology] = train_inds
        inds_test[pathology] = test_inds

        for info_k in ['u', 'u_prob', 'y', 'u_dist']:
            if info_k not in info_train:
                info_train[info_k] = {}
            if info_k not in info_test:
                info_test[info_k] = {}
            for k in [(pathology, 'pred'), (pathology, 'true')]:
                if info_k in info and k  in info[info_k]:
                    info_train[info_k][k] = index_iterable_with_ndarray(info[info_k][k], train_inds)
                    info_test[info_k][k] = index_iterable_with_ndarray(info[info_k][k], test_inds)
                
        if verbose:
            print(pathology)
            print('\t train: ', Counter(info_train['u'][(pathology, 'pred')]).most_common())
            print('\t test:  ', Counter(info_test['u'][(pathology, 'pred')]).most_common())

    return info_train, info_test, inds_train, inds_test



@dataclass
class FetchInfo:
    name: str
    keys_to_retrieve_value: List[tuple]
    modification_fn: Callable



def process_model_name_fn(x):
    if x.startswith('gpt-3.5-turbo'):
        return 'gpt-3.5-turbo'
    elif x.startswith('gpt-4o-mini'):
        return 'gpt-4o-mini'
    elif x.startswith('gpt-4o'):
        return 'gpt-4o'
    elif x.startswith('claude-3-5-sonnet'):
        return 'claude-3.5-sonnet'
    elif x.startswith('claude-3-haiku'):
        return 'claude-3-haiku'
    elif x.startswith('gemini-1.5-pro'):
        return 'gemini-1.5-pro'
    elif x.startswith('gemini-1.5-flash'):
        return 'gemini-1.5-flash'
    else:
        raise ValueError(f'Invalid model_name: {x}')

def process_task_name_fn(x):
    if x == 'sciq':
        return 'SciQ'
    elif x == 'truthfulqa':
        return 'TruthfulQA'
    else:
        raise ValueError(f'Invalid task_name: {x}')

def process_prompt_type_fn(x):
    if x == 'verbconf':
        s = 'Verb. Conf.'
    elif x == 'verbconfv2':
        s = 'Verb. Conf. (verbose)'
    elif x == 'verbconfv3':
        s = 'Verb. Dist.'
    elif x == 'verbprobtop1':
        s = 'Verb. Prob.'
    else:
        raise ValueError(f'Invalid prompt_type: {x}')
    return s

def process_certainty_phrase_source_fn(x):
    if x.startswith('prompt_'):
        kvs = parse_kv_from_string(x)
        model_name = kvs[1]
        size = kvs['size']
        s = f"prompt {model_name} " + f'($K={size}$)'
    elif x == 'social_media_poll':
        s = 'survey'
    elif x == 'unspecified_beta':
        s = 'on-the-fly'
    elif x == 'unspecified_delta':
        s = ''
    else:
        raise ValueError(f"Invalid certainty_phrase_source: {x}")
    return s


def set_calibrate_type_fn(data_subset):
    if data_subset == '':
        calibrate_type = ''
    elif data_subset == 'asverbtype':
        calibrate_type = ''
    elif 'platt' in data_subset:
        calibrate_type = 'Platt'
    elif 'histbinning' in data_subset:
        calibrate_type = 'Hist Binning'
    else:
        calibrate_type = 'OT'
    return calibrate_type


fetch_info_list = [
    FetchInfo('Path', [('save_dir',)], None),
    FetchInfo('Model', [('model_name',)], process_model_name_fn),
    FetchInfo('Task', [('task_name',)], process_task_name_fn),
    FetchInfo('Prompt', [('prompt_type',)], process_prompt_type_fn),
    FetchInfo(r'$(u_1,\cdots,u_K)$ Source', [('certainty_phrase_source',)], process_certainty_phrase_source_fn),
    FetchInfo('calibrate_type', [('data_subset',)], set_calibrate_type_fn),
]


def reconcile_ece_fn(ece_dist, ece_prob, prompt_type):
    if prompt_type.startswith('verbprob'):
        ece = ece_prob
    elif prompt_type.startswith('verbconf'):
        ece = ece_dist
    else:
        raise ValueError(f'Invalid prompt_type: {prompt_type}')
    return ece

group_metrics_by_task_metrics_cols = [
    'Correctness', 'MSE', 'ECE', 'AUC', 'BS', 'Accuracy', 'ece_prob', 'ece_dist']

fetch_info_list += [
    FetchInfo('Correctness', [('correctness',)], None),
    FetchInfo('MSE', [('mse',)], None),
    FetchInfo('ECE', [('ece_dist',), ('ece_prob',), ('prompt_type',)], reconcile_ece_fn),
    FetchInfo('AUC', [('auroc',)], None),
    FetchInfo('BS', [('brier_score',)], None),
    FetchInfo('Accuracy', [('accuracy',)], None),
]


def process_model_and_calibrated_fn(model_name, calibrate_type):
    if calibrate_type == '':
        s = model_name
    else:
        s = f"{model_name} calibrated ({calibrate_type})"
    return s

fetch_info_list += [
    FetchInfo('Model+calibrated', [('Model',), ('calibrate_type',)], process_model_and_calibrated_fn)
]



def write_to_jsonl(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')

class TaskResult:

    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.isdir(self.save_dir):
            raise ValueError(f"TaskResult at save_dir: {self.save_dir} does not exist on disk.")
        for k, v in self.load_config(self.save_dir).items():
            setattr(self, k, v)

    @cached_property
    def config_path(self):
        return os.path.join(self.save_dir, 'config.json')

    def copy_config_file(self, target_config_path, additional_config=None):
        if additional_config is None:
            shutil.copy(self.config_path, target_config_path)
        else:
            config = self.load_config(self.save_dir)
            for k, v in additional_config.items():
                config[k] = v
            with open(target_config_path, 'w') as f:
                json.dump(config, f)

    def load_config(self, save_dir):
        """Load from disk if available, otherwise parse from `save_dir` and save to disk. """
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        else:
            model_name = os.path.basename(os.path.dirname(save_dir))
            kvs = parse_kv_from_string(os.path.basename(save_dir))
            task_name = kvs[0]
            prompt_type = kvs['p']
            certainty_phrase_source = kvs.get('src', '')
            if certainty_phrase_source:
                certainty_phrase_source = certainty_phrase_source.replace(':', '_')
            if prompt_type == 'verbconfv3':
                assert(certainty_phrase_source == 'unspecified_beta')
            if certainty_phrase_source == '':
                certainty_phrase_source = 'unspecified_delta'
            config = {
                'model_name': model_name,
                'task_name': task_name,
                'prompt_type': prompt_type,
                'certainty_phrase_source': certainty_phrase_source,
            }
            with open(self.config_path, 'w') as f:
                json.dump(config, f)
        return config
        
    def __repr__(self):
        return f'TaskResult("{self.save_dir}")'

    @cached_property
    def certainty_phrase_distributions(self):
        return get_dcp_dist(self.certainty_phrase_source)

    @property
    def num_unique_certainty_phrases(self):
        return len(self.certainty_phrase_distributions) \
            if isinstance(self.certainty_phrase_distributions, dict) \
            else len(self.classification_data['u'][(self.model_name, 'pred')])

    @cached_property
    def classification_data(self):
        results_path = os.path.join(self.save_dir, 'answers_eval_correctness.jsonl')
        if self.certainty_phrase_source == 'unspecified_beta':
            fallback_confidence = 'Beta(1,1)'
        elif self.certainty_phrase_source == 'unspecified_delta':
            fallback_confidence = '0.5'
        else:
            confidence_phrases = list(self.certainty_phrase_distributions.keys())
            fallback_confidence = confidence_phrases[len(confidence_phrases)//2]
        info = get_llm_qa_response(
            model_name=self.model_name,
            file_path=results_path,
            dcp_dist=self.certainty_phrase_distributions,
            fallback_confidence=fallback_confidence,
        )
        return info

    @property
    def calibration_bin_data_prob_kwargs(self):
        num_bins = self.num_unique_certainty_phrases
        num_bins = 100
        calibration_kwargs = {
            'calibrate_over_dist': False,
            'calibration_type': 1,
            'num_bins': num_bins,
            'bootstrap_num_samples': 1,
            'bootstrap_alpha': 0.05,
            'include_cls_metrics': True,
        }
        return calibration_kwargs

    @cached_property
    def calibration_bin_data_prob(self):
        bin_data = compute_calibration_with_bootstrap(
            pathology=self.model_name,
            u_prob=self.classification_data['u_prob'],
            y=self.classification_data['y'],
            **self.calibration_bin_data_prob_kwargs,
        )
        return bin_data

    @property
    def calibration_bin_data_dist_kwargs(self):
        # don't compute calibration `bin_data` if all confidence are delta distributions. the resulting metrics are not meaningful anyways.
        if self.certainty_phrase_source == 'unspecified_delta':
            include_calibration_metrics = False,
        else:
            include_calibration_metrics = True
        num_bins = 100
        calibration_kwargs = {
            'calibrate_over_dist': True,
            'calibration_type': 1,
            'num_bins': num_bins,
            'bootstrap_num_samples': 100,
            'bootstrap_alpha': 0.05,
            'include_calibration_metrics': include_calibration_metrics,
            'include_cls_metrics': True,
            'integrate_dist': 'none', # just midpoint rule for all bins.
        }
        return calibration_kwargs

    @cached_property
    def calibration_bin_data_dist(self):
        bin_data = compute_calibration_with_bootstrap(
            pathology=self.model_name,
            u_dist=self.classification_data['u_dist'],
            **self.calibration_bin_data_dist_kwargs,
        )
        return bin_data

    @cached_property
    def ot_outputs(self):
        ot_outputs_filepath = os.path.join(self.save_dir, 'ot_outputs.pkl')
        if not os.path.isfile(ot_outputs_filepath):
            return {}
        else:
            with open(ot_outputs_filepath, 'rb') as f:
                return pickle.load(f)

    @cached_property
    def data_subset(self):
        basename = os.path.basename(self.save_dir)
        if any([basename.startswith(x) for x in ['train_', 'test_']]):
            return basename
        else:
            return ''

    @cached_property
    def answers(self):
        results_path = os.path.join(self.save_dir, 'answers_eval_correctness.jsonl')
        with open(results_path, 'r') as f:
            outputs = [json.loads(line) for line in f]
        return outputs

    def save_answers(self, subdir_name, outputs, force=True, additional_config=None):
        """ save `outputs`, having same format as `answers_eval_correctness.jsonl`,
            to `<self.save_dir>/<subdir_name>`. """
        jsonl_filepath = os.path.join(self.save_dir, subdir_name, 'answers_eval_correctness.jsonl')
        config_filepath = os.path.join(self.save_dir, subdir_name, 'config.json')
        # avoid overwriting answers if already exists.
        if not force and os.path.isfile(jsonl_filepath) and os.path.isfile(config_filepath):
            return
        write_to_jsonl(jsonl_filepath, outputs)
        self.copy_config_file(config_filepath, additional_config)

    def get_answers_list(self):
        """Get all answers subset under `save_dir`"""
        subdirs = {x: os.path.join(self.save_dir, x) for x in os.listdir(self.save_dir)}
        subdirs = {k: v for k, v in subdirs.items() if os.path.isdir(v)}
        rs = {k: TaskResult(v) for k,v in subdirs.items()}
        return rs

    def train_test_split(self, test_size=.5):
        """Create train/test split on model answer file. 
            save them to
            - train/answers_eval_correctness.jsonl
            - test/answers_eval_correctness.jsonl
        """
        data_train, data_test, inds_train, inds_test = train_test_split_pathology_labels_data(
            self.classification_data, test_size=test_size, verbose=False)
        inds_train = inds_train[self.model_name].tolist()
        inds_test = inds_test[self.model_name].tolist()

        outputs_train = [self.answers[ind] for ind in inds_train]
        outputs_test = [self.answers[ind] for ind in inds_test]

        self.save_answers('train', outputs_train, force=False)
        self.save_answers('test', outputs_test, force=False)

    @cached_property
    def metrics(self): 
        metrics = {}
        labels = self.classification_data['y'][(self.model_name, 'true')]
        metrics['correctness'] = labels.sum() / labels.size
        for k in ['auroc', 'mse', 'brier_score', 'accuracy']:
            metrics[k] = self.calibration_bin_data_dist[k]
        metrics['ece_prob'] = self.calibration_bin_data_prob['ece']
        if self.certainty_phrase_source == 'unspecified_delta':
            metrics['ece_dist'] = np.nan
        else:
            metrics['ece_dist'] = self.calibration_bin_data_dist['ece']
        metrics['data_subset'] = self.data_subset
        return metrics

    def get_df_subset(self, cols=None):
        df = self.df
        if cols is not None:
            df = df[[x for x in cols if x in df.columns]]
        return df

    @cached_property
    def df(self):

        data = {}
        for attr in ['save_dir', 'model_name', 'task_name', 'prompt_type', 'certainty_phrase_source', 'num_unique_certainty_phrases']:
            data[attr] = getattr(self, attr)
        data.update(self.metrics)

        for fetch_info in fetch_info_list:
            k = fetch_info.name
            vs = [dict_iterated_getitem(data, k) for k in fetch_info.keys_to_retrieve_value]
            if hasattr(fetch_info, 'modification_fn') and fetch_info.modification_fn is not None:
                v = fetch_info.modification_fn(*vs)
            else:
                if len(vs) != 1:
                    raise ValueError(f'If fetch_info.modification_fn is None, then should retrieve just one value but retrieved {len(vs)}')
                v = vs[0]
            data.update({k: v})

        df = pd.DataFrame.from_records([data])
        if df.isnull().all().all():
            return None
        return df


def get_eval_results(
        exp_dirs=None,
        run_paths=None,
        cols=None,
        run_path_filter_fn=None,
        group_metrics_by_task=True,
        sort_df=None,
        dry_run=False,
    ):

    if exp_dirs is None: exp_dirs = []
    if run_paths is None: run_paths = []
    if cols is None: cols = []

    info = {}
    info['df'] = pd.DataFrame()
    info['taskresults'] = None

    for exp_dir in exp_dirs:
        subdirs = list(os.listdir(exp_dir))
        if run_path_filter_fn is not None:
            subdirs = filter(run_path_filter_fn, subdirs)
        for subdir in subdirs:
            subdir_path = os.path.join(exp_dir, subdir)
            if not os.path.isfile(os.path.join(subdir_path, "answers_eval_correctness.jsonl")):
                continue
            if subdir_path not in run_paths:
                run_paths.append(subdir_path)

    info['run_paths'] = run_paths

    if dry_run is True:
        return info
    
    dfs = []
    runs = []
    for run_path in run_paths:
        r = TaskResult(run_path)
        df = r.get_df_subset(cols)
        if df is None:
            continue
        dfs.append(df)
        runs.append(r)

        if dry_run == 'fast':
            break

    filtered_dfs = [df.dropna(axis=1, how="all") for df in dfs]  # remove nan cols
    df = pd.concat(filtered_dfs, axis=0, ignore_index=True)

    non_numeric_columns = df.select_dtypes(exclude=[np.number])
    df[non_numeric_columns.columns] = non_numeric_columns.fillna('')

    if group_metrics_by_task:
        if 'Task' not in df.columns:
            raise ValueError(f'Require `Task` as a column but get cols: {cols}')
        # figure out the values for the pivot
        df_cols = list(df.columns)
        metric_names = set(df_cols).intersection(set(group_metrics_by_task_metrics_cols))
        metric_names = sorted(metric_names, key=lambda x: cols.index(x))
        # figure out the rest of the columns as index for the pivot
        rest_cols = set(df_cols) - set(group_metrics_by_task_metrics_cols + ['Task'])
        rest_cols = sorted(rest_cols, key=lambda x: cols.index(x))
        # pivot
        dfp = df.pivot(index=rest_cols, columns='Task', values=metric_names)
        # swap Metrics/Task -> Task/Metrics
        dfp.columns = dfp.columns.swaplevel(0, 1)
        ## resort the columns to follow the ordering specified in `cols`
        dfp = dfp.sort_index(axis=1)
        # x[0] top level {Model, SciQ, TruthfulQA}, x[1] next level {BS,ECE,etc.}
        # sort by top level, then next level
        cols_reorder = list(dfp.columns).copy()
        tasks = list(dict.fromkeys([x[0] for x in cols_reorder if x[1] != '']))
        def ordering_fn(x):
            toplevel_key = (tasks.index(x[0])+1)*100 if x[0] in tasks else 0
            nexlevel_key = cols.index(x[1] if x[0] in tasks else x[0])
            return toplevel_key + nexlevel_key
        cols_reorder = sorted(cols_reorder, key=ordering_fn)
        dfp = dfp.reindex(columns=cols_reorder)
        ##
        dfp = dfp.reset_index()
        df = dfp

    if sort_df:
        ranks = df.select_dtypes(include=['number']).rank(ascending=False)
        df['Ranking'] = ranks.fillna(ranks.mean()).mean(axis=1)
        df_metrics = df.select_dtypes(include=['number']).drop(columns=[x for x in ['Ranking', 'Metrics/Avg'] if x in df.columns])
        df['Metrics/Avg'] = df_metrics.fillna(df_metrics.mean()).mean(axis=1)
        if sort_df.startswith('mean'):
            df = df.sort_values(by=['Metrics/Avg'], ascending='ascending' in sort_df)
            df = df.drop(columns=['Ranking'])
        elif sort_df.startswith('ranking'):
            df = df.sort_values(by=['Ranking'], ascending='ascending' in sort_df)
            df = df.drop(columns=['Metrics/Avg'])
        else:
            raise ValueError(f'Invalid sort_df: {sort_df}')

    info['df'] = df
    info['taskresults'] = runs

    return info
        


def jpt_default_display(df):
    from IPython.display import display
    display(df
        .style
        .set_table_styles([{'selector': 'td', 'props': [('white-space', 'pre-wrap'), ('word-wrap', 'break-word')]},
                        {'selector': 'th', 'props': [('border-bottom', '2px solid black')]}])
        .set_properties(**{'text-align': 'left'})
        .background_gradient(cmap='RdYlGn_r', subset=[col for col in df.select_dtypes(include=[ np.number]).columns if col != 'Ranking'], low=.2, high=.8)
        .background_gradient(cmap='RdYlGn_r', subset=['Ranking'] if 'Ranking' in df else [], low=.2, high=.8)
        .format({k: '{:.2f}' if k in [('Metrics', 'Avg')] else '{:.3f}' for k in df.select_dtypes(include=[np.number]).columns})
    )


