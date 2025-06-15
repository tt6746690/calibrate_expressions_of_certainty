import torch
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    mean_squared_error,
    brier_score_loss)


def parse_kv_from_string(s):
    kvs = []
    for i, segment in enumerate(s.split('_')):
        if '=' in segment:
            k, v = segment.split('=', 1)
            try:
                if '.' in v:
                    v = float(v)
                else:
                    v = float(v)
                    if v.is_integer():
                        v = int(v)
            except:
                pass
            kvs.append((k, v))
        else:
            kvs.append((i, segment))
    d = dict(kvs)
    return d


def dict_iterated_getitem(d, ks):
    """ Get dictionary item successively.
        
        `ks`
            ['k1','k2'] or 'k1.k2'
    """
    if isinstance(ks, str):
        ks = ks.split('.')
    x = d
    for k in ks:
        x = x[k]
    return x     


def joblib_parallel_process(fn, iterable, n_jobs, prefer=None, use_tqdm=False):
    """Computes [fn(x) for x in iterable] with `n_jobs` number of processes.
            Setting `use_tqdm` to True implicitly converts `iterable` to list.

        Sometimes setting `backend='loky'` makes too many cores running.
            `backend='multiprocessing'` seems to be less CPU compute intensive.
    """
    parallel_execute = Parallel(n_jobs=n_jobs,
                                prefer=prefer)
    delayed_fn = delayed(fn)
    if use_tqdm:
        from tqdm import tqdm
        desc = use_tqdm if isinstance(use_tqdm, str) else ""
        try:
            iterable = tqdm(iterable, total=len(iterable), desc=desc)
        except TypeError:
            iterable = tqdm(list(iterable), desc=desc)
    result = parallel_execute(delayed_fn(x) for x in iterable)
    return result



def torch_tensor_to_ndarray(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().to('cpu').numpy()
    return x

def log_loss_fp32(label, score, **kwargs):
    """When score is `float32`, computing `log(score)` will introduce
            extreme values """
    return log_loss(label, score.astype(np.float64), **kwargs)


def metrics_binary_classification(label, score, threshold=.5, nll_class_weights=None):
    """Metrics for binary classification.
            label           (n_samples,)
            score           (n_samples,)
                after application of sigmoid
    """
    label = torch_tensor_to_ndarray(label)
    score = torch_tensor_to_ndarray(score)
    nll_class_weights = torch_tensor_to_ndarray(nll_class_weights)

    pred = (score > threshold).astype(np.int32)

    metrics = {}
    metrics['N'] = len(label)
    metrics['nll'] = log_loss_fp32(label, score, labels=np.arange(2))
    if nll_class_weights is not None:
        metrics['nll_weighted'] = log_loss_fp32(
            label, score, sample_weight=nll_class_weights[label.astype(np.int)])
    metrics['accuracy'] = accuracy_score(label, pred)
    metrics['precision'], metrics['recall'], metrics['f1_score'], _ = precision_recall_fscore_support(
        label, pred, average='macro', zero_division=0)
    metrics['precision_avg'] = average_precision_score(
        label, score, average='macro')
    try:
        metrics['auroc'] = roc_auc_score(label, score)
    except:
        metrics['auroc'] = 0.
    metrics['mse'] = mean_squared_error(label, pred)
    metrics['brier_score'] = brier_score_loss(label, score)

    return metrics



def plt_scaled_colobar_ax(ax):
    """ Create color bar
            fig.colorbar(im, cax=plt_scaled_colobar_ax(ax)) 
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7.5%", pad=0.05)
    return cax


def plt_kernel_matrix_one(fig, ax, K, title=None, n_ticks=5,
                          custom_ticks=True, vmin=None, vmax=None, annotate=False, cmap='jet'):
    im = ax.imshow(K, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title if title is not None else '')
    fig.colorbar(im, cax=plt_scaled_colobar_ax(ax))
    # custom ticks
    if custom_ticks:
        n = len(K)
        ticks = list(range(n))
        ticks_idx = np.rint(np.linspace(
            1, len(ticks), num=min(n_ticks,    len(ticks)))-1).astype(int)
        ticks = list(np.array(ticks)[ticks_idx])
        ax.set_xticks(np.linspace(0, n-1, len(ticks)))
        ax.set_yticks(np.linspace(0, n-1, len(ticks)))
        ax.set_xticklabels(ticks)
        ax.set_yticklabels(ticks)
    if annotate:
        annotate_str_template = annotate if isinstance(annotate, str) else '{:.2f}'
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                ax.annotate(annotate_str_template.format(K[i,j]), xy=(j, i),
                            horizontalalignment='center',
                            verticalalignment='center')
    return fig, ax