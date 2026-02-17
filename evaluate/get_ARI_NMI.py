import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
import numpy as np
import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting sparse tensor to CSR format for more efficient processing.*",
)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def kmeans_ari_nmi_stats(Z, labels, K=None, runs=50, seeds=None, n_init=1):
    """
    Run KMeans 'runs' times and return mean/std for ARI & NMI.
    """
    if K is None:
        K = len(np.unique(labels))
    if seeds is None:
        rng = np.random.RandomState(0)
        seeds = rng.randint(0, 2**31 - 1, size=runs)

    aris, nmis = [], []
    for s in seeds:
        km = KMeans(n_clusters=K, n_init=n_init, random_state=int(s))
        preds = km.fit_predict(Z)
        aris.append(adjusted_rand_score(labels, preds))
        nmis.append(normalized_mutual_info_score(labels, preds))

    aris = np.asarray(aris, dtype=float)
    nmis = np.asarray(nmis, dtype=float)
    return aris.mean(), aris.std(ddof=1), nmis.mean(), nmis.std(ddof=1)


def kmeans_ari_nmi_mean(Z, labels, K=None, n_runs=50, n_init=20):
    if K is None:
        K = len(np.unique(labels))

    ari_scores = []
    nmi_scores = []

    for seed in range(n_runs):
        km = KMeans(n_clusters=K, n_init=n_init, random_state=seed)
        preds = km.fit_predict(Z)
        ari_scores.append(adjusted_rand_score(labels, preds))
        nmi_scores.append(normalized_mutual_info_score(labels, preds))

    return (
        np.mean(ari_scores),
        np.mean(nmi_scores),
    )
