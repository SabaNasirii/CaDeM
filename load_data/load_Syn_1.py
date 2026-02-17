import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting sparse tensor to CSR format for more efficient processing.*",
)

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from process_data.data_processing import *


def generate_distinct_labels(n_nodes, n_communities, seed=42):
    """Randomly assign community labels to nodes."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_communities, size=n_nodes)


def generate_adjacency(labels, p_intra=0.7, p_inter=0.1, seed=None):
    """Generate adjacency using SBM."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            p = p_intra if labels[i] == labels[j] else p_inter
            if rng.rand() < p:
                A[i, j] = A[j, i] = 1
    np.fill_diagonal(A, 0)
    return A.astype(np.int64)


def choose_final_labels(view_labels, probs, n_nodes, seed=None):
    """
    Select final labels based on given probabilities.
    probs = [prob_view1, prob_view2, prob_view3,...]
    """
    rng = np.random.RandomState(seed)
    probs = np.array(probs) / np.sum(probs)
    final_labels = np.zeros(n_nodes, dtype=int)
    for i in range(n_nodes):
        choice = rng.choice(len(probs), p=probs)
        final_labels[i] = view_labels[choice][i]

    return final_labels


def load_Syn1(
    n_nodes=100,
    n_graphs=3,
    n_communities=3,
    p_intra=0.7,
    p_inter=0.1,
    label_probs=[0.3, 0.3, 0.3],
):
    seed = 42
    labels_views = [
        generate_distinct_labels(n_nodes, n_communities, seed=seed + 100 * v)
        for v in range(n_graphs)
    ]
    adj_matrices = [
        generate_adjacency(labels_views[v], p_intra, p_inter, seed=seed + 200 * v)
        for v in range(n_graphs)
    ]
    adj_mats = []
    for i in adj_matrices:
        adj_mats.append(csr_matrix(i))

    final_labels = choose_final_labels(
        labels_views, label_probs, n_nodes=n_nodes, seed=seed + 7000
    )

    feat_mats = sp.eye(n_nodes)
    feat_mats = torch.FloatTensor(preprocess_features_Syn1(feat_mats))
    community_assignments = torch.tensor(onehot_encoding(final_labels)).float()

    return adj_mats, feat_mats, community_assignments, labels_views
