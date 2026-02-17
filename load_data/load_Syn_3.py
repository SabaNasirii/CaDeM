import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
import numpy as np
import scipy.sparse as sp
import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting sparse tensor to CSR format for more efficient processing.*",
)
from typing import Tuple, Dict, List, Optional
import numpy.linalg as npla
from sklearn.neighbors import NearestNeighbors
import math
from numpy.linalg import eigh
from collections import deque
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from process_data.data_processing import *


def spectral_radius(
    A: sp.spmatrix,
    iters: int = 100,
    tol: float = 1e-6,
    rs: Optional[np.random.RandomState] = None,
) -> float:
    """
    Power iteration to estimate largest eigenvalue of symmetric PSD matrix.
    """
    n = A.shape[0]
    if rs is None:
        rs = np.random.RandomState(0)
    v = rs.randn(n)
    v /= npla.norm(v) + 1e-12
    lam_old = 0.0
    for _ in range(iters):
        w = A @ v
        lam = float(v @ w)
        nrm = npla.norm(w) + 1e-12
        v = w / nrm
        if abs(lam - lam_old) < tol * (abs(lam_old) + 1e-12):
            break
        lam_old = lam
    return max(lam, 1e-12)


def graph_knn(
    X: np.ndarray, k: int = 12, sigma: Optional[float] = None
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Build symmetric kNN graph.
    Returns adjacency (csr) and pairwise distances (only for neighbors).
    """
    N = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    dist, idx = nbrs.kneighbors(X)
    if sigma is None:
        sig = np.median(dist[:, 1:].ravel())
        sigma = sig if sig > 1e-12 else 1.0
    rows, cols, data = [], [], []
    for i in range(N):
        for j_idx in range(1, k + 1):
            j = idx[i, j_idx]
            d2 = dist[i, j_idx] ** 2
            w = math.exp(-d2 / (sigma**2))
            rows.append(i)
            cols.append(j)
            data.append(w)
            rows.append(j)
            cols.append(i)
            data.append(w)
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    A = 0.5 * (A + A.T)
    return A, dist


def laplacian(
    A: sp.csr_matrix, normalized: bool = False
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Graph Laplacian L or normalized Lsym.
    Returns L and degree vector d.
    """
    d = np.asarray(A.sum(axis=1)).ravel()
    if not normalized:
        L = sp.diags(d) - A
    else:
        with np.errstate(divide="ignore"):
            d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        L = sp.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return L.tocsr(), d


def kernel_lowpass_h(lmbda: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Smooth low-pass: h(λ) = exp(-λ / α). α controls cutoff softness.
    """
    return np.exp(-lmbda / max(alpha, 1e-12))


def bandpass_sharp(u):
    return u * np.exp(-((u) ** 2))


def lanczos_tridiag(A, v, m):
    n = A.shape[0]
    V = np.zeros((n, m))
    alpha = np.zeros(m)
    beta = np.zeros(m - 1)
    v0 = v / (np.linalg.norm(v) + 1e-12)
    V[:, 0] = v0
    w = A @ v0
    alpha[0] = float(v0 @ w)
    r = w - alpha[0] * v0
    m_eff = 1
    for j in range(1, m):
        beta[j - 1] = float(np.linalg.norm(r))
        if beta[j - 1] < 1e-14:
            break
        V[:, j] = r / beta[j - 1]
        w = A @ V[:, j]
        alpha[j] = float(V[:, j] @ w)
        r = w - alpha[j] * V[:, j] - beta[j - 1] * V[:, j - 1]
        m_eff += 1
    return alpha[:m_eff], beta[: max(0, m_eff - 1)], V[:, :m_eff]


def f_of_L_times_v_lanczos(L, v, g_fun, m=100):
    nrm = np.linalg.norm(v)
    if nrm < 1e-15:
        return np.zeros_like(v)
    alpha, beta, V = lanczos_tridiag(L, v / nrm, m)
    T = np.diag(alpha)
    if len(beta) > 0:
        off = np.diag(beta, 1)
        T = T + off + off.T
    th, Q = eigh(T)
    e1 = np.zeros(len(th))
    e1[0] = 1.0
    gth = g_fun(th)
    z = Q @ (gth * (Q.T @ e1))
    return (V @ z) * nrm


def atom_psi_lanczos(L, center, t, g_fun=lambda u: u * np.exp(-u), m=100):
    n = L.shape[0]
    delta = np.zeros(n)
    delta[center] = 1.0
    return f_of_L_times_v_lanczos(L, delta, g_fun=lambda lam: g_fun(t * lam), m=m)


def bfs_hop_distances(A, src, R):
    """
    Unweighted BFS up to R hops from src.
    Returns: (order, dist) where:
      - order: list of visited nodes in discovery order
      - dist:  float array of hop distances (inf for unreachable)
    """
    B = A.tocsr().copy()
    B.data[:] = 1.0
    n = A.shape[0]
    dist = np.full(n, np.inf)
    dist[src] = 0.0
    q = deque([src])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        if dist[u] >= R:
            continue
        nbrs = B.indices[B.indptr[u] : B.indptr[u + 1]]
        for v in nbrs:
            if dist[v] == np.inf:
                dist[v] = dist[u] + 1.0
                q.append(v)
    return order, dist


def cosine_taper_by_hops(dist_vals, R, avoid_center=False):
    """
    Cosine weights by hop distance in [0..R]:
      w(d) = 0.5 * (1 + cos(pi * d / R)), truncated to 0 outside [0,R].
    If avoid_center=True, set weight at d=0 to 0 to suppress the exact center spike.
    """
    d = np.asarray(dist_vals, float)
    w = 0.5 * (1.0 + np.cos(np.pi * np.clip(d, 0, R) / max(R, 1)))
    w[d > R] = 0.0
    if avoid_center:
        w[d == 0.0] = 0.0
    return w


def add_pattern_around_hotspot(
    L,
    A,
    centers,
    t,
    R,
    stride=1,
    per_center_norm=1.0,
    avoid_center=False,
    g_fun=lambda u: u * np.exp(-u),
    m=120,
):
    """
    Build a private pattern as a weighted sum of atoms placed on each center's R-hop neighborhood.
    For each center c:
      - BFS to get nodes up to R hops
      - Select nodes with step=stride
      - Cosine-taper weights by hop distance
      - Normalize weights to sum to 1
      - Sum wavelet atoms ψ_{t,node} weighted by w
      - Normalize the per-center blob to 'per_center_norm'
    Finally, sum blobs over all centers.

    Returns:
      xP1: ndarray (N,), the private signal.
    """
    n = L.shape[0]
    x = np.zeros(n)

    for c in centers:
        order, dist = bfs_hop_distances(A, c, R)
        sel = order[:: max(1, stride)]
        dsel = dist[sel]

        # cosine weights by hop distance
        w = cosine_taper_by_hops(dsel, R=R, avoid_center=avoid_center)
        if w.sum() <= 1e-12:
            continue
        w = w / w.sum()

        # accumulate atoms around center c
        xc = np.zeros(n)
        for node, ww in zip(sel, w):
            # atom ψ_{t,node} = g(tL) δ_node  (Lanczos)
            delta = np.zeros(n)
            delta[node] = 1.0
            psi = f_of_L_times_v_lanczos(
                L, delta, g_fun=lambda lam: g_fun(t * lam), m=m
            )
            xc += ww * psi

        # normalize per-center blob to requested norm
        xc /= np.linalg.norm(xc)
        xc *= per_center_norm

        x += xc

    return x


def geodesic_knn_dist(A: sp.csr_matrix, sources: List[int], r: int = 2) -> np.ndarray:
    """
    graph-radius mask: select nodes within <= r hops from nearest source.
    Uses BFS on unweighted graph; appropriate to extract local labels around node centers.
    """
    n = A.shape[0]
    B = A.copy()
    B.data[:] = 1.0
    dist = np.full(n, np.inf)
    q = deque()
    for s in sources:
        dist[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        if dist[u] >= r:
            continue
        start = B.indptr[u]
        end = B.indptr[u + 1]
        nbrs = B.indices[start:end]
        for v in nbrs:
            if dist[v] == np.inf:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def label_assignment(
    A: sp.csr_matrix, centers: List[int], r_hops: int = 2, keep_background: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Assign each node to nearest hot-spot center if within r hops; else background.
    """
    n = A.shape[0]
    dist = geodesic_knn_dist(A, centers, r=r_hops)
    y = np.full(n, len(centers), dtype=int)
    for v in range(n):
        if dist[v] <= r_hops:
            nearest = np.argmin([abs(v - c) for c in centers])
            y[v] = nearest
    K_eff = len(centers) + (1 if keep_background else 0)
    if not keep_background:
        y[y == len(centers)] = 0
        K_eff = len(centers)
    return y, K_eff


def pick_centers(X, m=3, outer_quantile=0.7, angle_pad_deg=8.0, seed=123):
    """
    Place m centers along the spiral.

    X: (N,2) point coordinates
    """
    N = X.shape[0]
    theta = np.arctan2(X[:, 1], X[:, 0])
    theta = np.unwrap(theta)
    r = np.linalg.norm(X, axis=1)

    thr = np.quantile(r, outer_quantile)
    cand = np.where(r >= thr)[0]

    th_min, th_max = theta[cand].min(), theta[cand].max()
    sector_edges = np.linspace(th_min, th_max, m + 1)
    sector_centers = 0.5 * (sector_edges[:-1] + sector_edges[1:])

    chosen = []
    used_mask = np.zeros(N, dtype=bool)

    pad = np.deg2rad(angle_pad_deg)

    for thc in sector_centers:
        idx = cand[~used_mask[cand]]

        if idx.size == 0:
            break

        dth = np.abs(theta[idx] - thc)
        K = max(10, int(0.05 * idx.size))
        pool = idx[np.argsort(dth)[:K]]
        pick = pool[np.argmax(r[pool])]
        chosen.append(int(pick))

        close = np.abs(theta[cand] - theta[pick]) <= pad
        used_mask[cand[close]] = True

    while len(chosen) < m:
        idx = cand[~used_mask[cand]]
        if idx.size == 0:
            break
        if chosen:
            dmin = np.min(
                np.abs(theta[idx][:, None] - theta[np.array(chosen)][None, :]), axis=1
            )
        else:
            dmin = np.ones(idx.size) * 1e9
        pick = idx[np.argmax(dmin)]
        chosen.append(int(pick))
        close = np.abs(theta[cand] - theta[pick]) <= pad
        used_mask[cand[close]] = True

    return chosen[:m]


def build_geometry_spiral(
    N: int = 500, noise: float = 0.01, rs: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    2D spiral-ish manifold. Returns positions X[N,2]
    """
    if rs is None:
        rs = np.random.RandomState(0)
    angles = np.linspace(2.0 * np.pi, 6.0 * np.pi, N) + rs.randn(N) * 0.03
    radii = np.linspace(0.1, 1.0, N)
    X = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    X += rs.randn(N, 2) * noise
    return X


def one_time_setup(
    N=500, k=12, graph_sigma=None, normalized=False, J=6, alpha_h=1.0, seed=123
):
    """
    Build geometry, graph and Laplacian.
    """
    rs = np.random.RandomState(seed)
    np.random.seed(seed)

    # Geometry and graph
    X = build_geometry_spiral(N=N, rs=rs)
    A, _ = graph_knn(X, k=k, sigma=graph_sigma)
    L, deg = laplacian(A, normalized=normalized)

    # Lanczos path
    lmax = spectral_radius(L, iters=200, rs=rs)
    lmax = max(float(lmax), 1e-6)
    ts = np.geomspace(1.0 / lmax, 20.0 / lmax, J)

    setup = {
        "X": X,
        "A": A,
        "L": L,
        "deg": deg,
        "filt": {"lmax": lmax, "ts": ts, "alpha_h": alpha_h},
        "rs": rs,
    }
    return setup


def generate_multiplex_graph(
    setup: Dict,
    centers_P1: List[int],
    centers_P2: List[int],
    snr_common: float = 0.8,
    snr_private: float = 1.0,
) -> Dict:

    L = setup["L"]
    rs = setup["rs"]
    n = L.shape[0]
    ts = setup["filt"]["ts"]
    alpha_h = setup["filt"]["alpha_h"]

    # Low-pass common signal via Lanczos
    z = rs.randn(n)
    x_common = f_of_L_times_v_lanczos(
        L, z, g_fun=lambda lam: kernel_lowpass_h(lam, alpha=alpha_h), m=100
    )
    x_common = x_common / (np.linalg.norm(x_common)) * snr_common

    # P1 private (Layer 1)
    ts = setup["filt"]["ts"]
    t_hot = ts[5]

    xP1 = add_pattern_around_hotspot(
        L=setup["L"],
        A=setup["A"],
        centers=centers_P1,
        t=t_hot,
        R=3,
        stride=1,
        per_center_norm=1.0,
        avoid_center=False,
        g_fun=lambda u: bandpass_sharp(u),  # or kernel_bandpass_g
        m=120,
    )
    xP1 = xP1 / (np.linalg.norm(xP1)) * snr_private

    # P2 private (Layer 2)
    ts = setup["filt"]["ts"]
    t_hot = ts[5]

    xP2 = add_pattern_around_hotspot(
        L=setup["L"],
        A=setup["A"],
        centers=centers_P2,
        t=t_hot,
        R=2,
        stride=1,
        per_center_norm=1.0,
        avoid_center=False,
        g_fun=lambda u: bandpass_sharp(u),  # or kernel_bandpass_g
        m=120,
    )
    xP2 = xP2 / (np.linalg.norm(xP2) + 1e-12) * snr_private

    x1 = 0.3 * x_common + xP1
    x2 = 0.3 * x_common + xP2

    layers = {
        "x_common": x_common,
        "xP1": xP1,
        "xP2": xP2,
        "layer1": x1,
        "layer2": x2,
    }
    return layers


def load_Syn3(
    N=600,
    k=14,
    m=3,
    normalized=False,
    J=6,
    alpha_h=1.0,
    snr_common=0.8,
    snr_private=1.0,
):
    seed = 123
    setup = one_time_setup(
        N=N, k=k, normalized=normalized, J=J, alpha_h=alpha_h, seed=seed
    )
    X = setup["X"]
    N = X.shape[0]

    centers_P1 = pick_centers(X, m=m, outer_quantile=0.5, angle_pad_deg=20.0, seed=seed)
    centers_P2 = pick_centers(X, m=m, outer_quantile=0, angle_pad_deg=100, seed=seed)

    layers = generate_multiplex_graph(
        setup,
        centers_P1=centers_P1,
        centers_P2=centers_P2,
        snr_common=snr_common,
        snr_private=snr_private,
    )

    yP1, _ = label_assignment(setup["A"], centers_P1, r_hops=2, keep_background=True)
    yP2, _ = label_assignment(setup["A"], centers_P2, r_hops=2, keep_background=True)

    A = setup["A"]
    B = A.tocsr().copy()
    B.data[:] = 1.0

    adj_mats = []
    adj_mats.append(B)
    adj_mats.append(B)

    feat_mats = [
        torch.FloatTensor(layers["layer1"].reshape(N, -1)),
        torch.FloatTensor(layers["layer2"].reshape(N, -1)),
    ]

    labels_1 = torch.tensor(onehot_encoding(yP1)).float()
    labels_2 = torch.tensor(onehot_encoding(yP2)).float()

    return adj_mats, feat_mats, labels_1, labels_2
