import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
import numpy as np
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting sparse tensor to CSR format for more efficient processing.*",
)
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from process_data.data_processing import *


def generate_sbm_multiplex(n, K, L, pin, pout, pin_priv, pout_priv, lam, seed, portion):
    """
    Build a multiplex SBM where per-layer edge probabilities are a convex combination
    of a SHARED block model (P0) and a LAYER-SPECIFIC one (Q^(ℓ)):

      P_ij^(ℓ) = λ * P0_{c_i,c_j} + (1-λ) * Q^(ℓ)_{s_i^(ℓ), s_j^(ℓ)}

    Returns:
      As        : list of (n,n) adjacency matrices for L layers (0/1, undirected)
      C         : (n,) shared community labels
      S_layers  : list of (n,) private community labels per layer
    """
    rng = np.random.default_rng(seed)

    # Shared labels and block matrix
    C = rng.integers(0, K, size=n)
    P0 = pout * np.ones((K, K))
    np.fill_diagonal(P0, pin)

    As, S_layers = [], []
    for ell in range(L):
        # Private labels = perturbation of C
        S = C.copy()
        flip_idx = rng.choice(n, size=int(portion * n), replace=False)
        S[flip_idx] = rng.integers(0, K, size=len(flip_idx))
        S_layers.append(S)

        # Private block matrix
        Q = pout_priv * np.ones((K, K))
        np.fill_diagonal(Q, pin_priv)

        # Node-pair probability matrix (mixed at node level)
        P_shared = P0[C][:, C]
        P_priv = Q[S][:, S]
        P = lam * P_shared + (1.0 - lam) * P_priv
        P = np.clip(P, 1e-6, 1 - 1e-6)
        np.fill_diagonal(P, 0.0)

        # Sample undirected graph
        U = rng.uniform(size=(n, n))
        A = (U < P).astype(int)
        A = np.triu(A, 1)
        A = A + A.T
        As.append(A)

    return As, C, S_layers


def load_Syn2(n, K, L, pin, pout, pin_priv, pout_priv, lam, seed, portion):
    adj_matrices, C, S_layers = generate_sbm_multiplex(
        n=n,
        K=K,
        L=L,
        pin=pin,
        pout=pout,
        pin_priv=pin_priv,
        pout_priv=pout_priv,
        lam=lam,
        seed=seed,
        portion=portion,
    )

    adj_mats = []
    for i in adj_matrices:
        adj_mats.append(csr_matrix(i))

    feat_mats = np.eye(n)
    feat_mats = torch.FloatTensor(preprocess_features(feat_mats))
    common_labels = torch.tensor(onehot_encoding(C)).float()
    for i in range(len(S_layers)):
        S_layers[i] = torch.tensor(onehot_encoding(S_layers[i])).float()

    return adj_mats, feat_mats, common_labels, S_layers
