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
from typing import List
from scipy.sparse.linalg import eigs
import networkx as nx
from dataclasses import dataclass
from dataclasses import replace


def normalize_by_spectral_radius(A: csr_matrix) -> csr_matrix:
    """Normalize adjacency by its spectral radius."""
    if A.nnz == 0:
        return A.copy()
    try:
        vals = eigs(
            A.asfptype(), k=1, which="LM", return_eigenvectors=False, maxiter=1000
        )
        lam_max = np.abs(vals[0].real)
    except Exception:
        lam_max = max(A.shape)
    if lam_max <= 1e-9:
        return A.copy()
    return (A / lam_max).tocsr()


def to_csr(G: nx.Graph) -> csr_matrix:
    return nx.to_scipy_sparse_array(G, format="csr", dtype=float)


@dataclass
class DynParams:
    beta: float = 1.0
    steps: int = 200
    dt: float = 0.05
    noise: float = 0.0
    seed: int = 0


def simulate_population(A: csr_matrix, x0: np.ndarray, p: DynParams) -> np.ndarray:
    rng = np.random.default_rng(p.seed)
    A_scaled = normalize_by_spectral_radius(A) * p.beta
    x = x0.copy()
    for _ in range(p.steps):
        x2 = x * x
        dx = -(x**3) + A_scaled.dot(x2)
        if p.noise:
            dx += p.noise * rng.normal(size=x.shape)
        x += p.dt * dx
    return x


def simulate_epidemic(A: csr_matrix, x0: np.ndarray, p: DynParams) -> np.ndarray:
    rng = np.random.default_rng(p.seed)
    A_scaled = normalize_by_spectral_radius(A) * p.beta
    x = np.clip(x0.copy(), 0.0, 1.0)
    for _ in range(p.steps):
        dx = -x + A_scaled.dot((1.0 - x) * x)
        if p.noise:
            dx += p.noise * rng.normal(size=x.shape)
        x = np.clip(x + p.dt * dx, 0.0, 1.0)
    return x


def simulate_mutualistic(A: csr_matrix, x0: np.ndarray, p: DynParams) -> np.ndarray:
    rng = np.random.default_rng(p.seed)
    A_scaled = normalize_by_spectral_radius(A) * p.beta
    x = np.clip(x0.copy(), 0.0, 1.0)
    for _ in range(p.steps):
        sat = (x * x) / (1.0 + x * x)
        dx = x * (1.0 - x) + A_scaled.dot(x * sat)
        if p.noise:
            dx += p.noise * rng.normal(size=x.shape)
        x = np.clip(x + p.dt * dx, 0.0, 1.0)
    return x


def simulate_regulatory_R1(A: csr_matrix, x0: np.ndarray, p: DynParams) -> np.ndarray:
    rng = np.random.default_rng(p.seed)
    A_scaled = normalize_by_spectral_radius(A) * p.beta
    x = np.clip(x0.copy(), 0.0, None)
    for _ in range(p.steps):
        term = np.power(np.abs(x), 1 / 3)
        dx = -x + A_scaled.dot(term / (1.0 + term))
        if p.noise:
            dx += p.noise * rng.normal(size=x.shape)
        x += p.dt * dx
    return x


def simulate_biochemical(A: csr_matrix, x0: np.ndarray, p: DynParams) -> np.ndarray:
    rng = np.random.default_rng(p.seed)
    A_scaled = normalize_by_spectral_radius(A) * p.beta
    x = x0.copy()
    for _ in range(p.steps):
        dx = 1.0 - x - A_scaled.dot(x * x)
        if p.noise:
            dx += p.noise * rng.normal(size=x.shape)
        x += p.dt * dx
    return x


def simulate_regulatory_R2(A: csr_matrix, x0: np.ndarray, p: DynParams) -> np.ndarray:
    rng = np.random.default_rng(p.seed)
    A_scaled = normalize_by_spectral_radius(A) * p.beta
    x = np.clip(x0.copy(), 0.0, 1.0)
    for _ in range(p.steps):
        frac = (x**2) / (1.0 + x**2)
        dx = -x + A_scaled.dot(frac)
        if p.noise:
            dx += p.noise * rng.normal(size=x.shape)
        x = np.clip(x + p.dt * dx, 0.0, 1.0)
    return x


DYNAMICS = {
    "population": simulate_population,
    "regulatory_R1": simulate_regulatory_R1,
    "epidemic": simulate_epidemic,
    "biochemical": simulate_biochemical,
    "mutualistic": simulate_mutualistic,
    "regulatory_R2": simulate_regulatory_R2,
}


@dataclass
class MultiplexGraph:
    A1: csr_matrix
    A2: csr_matrix
    xL1: np.ndarray
    xL2: np.ndarray
    base_id: int
    label: int


def random_initial_states(n: int, regime: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if regime == "population":
        return rng.normal(0.0, 0.5, size=n)
    elif regime == "regulatory_R1":
        return rng.uniform(0, 0.5, size=n)

    elif regime == "epidemic" or regime == "biochemical":
        return rng.uniform(0.0, 0.4, size=n)
    elif regime == "biochemical":
        return rng.beta(2, 2, size=n)

    elif regime == "mutualistic":
        return rng.uniform(0.0, 0.6, size=n)
    elif regime == "regulatory_R2":
        return rng.uniform(0.0, 0.6, size=n)

    else:
        raise ValueError("Unknown regime")


def generate_base_graph(n: int, p_edge: float, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n=n, p=p_edge, seed=int(rng.integers(1e9)))
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for c1, c2 in zip(comps[:-1], comps[1:]):
            u = rng.choice(list(c1))
            v = rng.choice(list(c2))
            G.add_edge(int(u), int(v))
    return G


def build_dataset(
    n_nodes: int = 100,
    p_edge: float = 0.08,
    n_structures: int = 30,
    global_seed: int = 42,
) -> List[MultiplexGraph]:
    """Create 3 classes * n_structures multiplex graphs."""

    dyn_params = {
        "population": DynParams(beta=1.0, steps=250, dt=0.04, seed=global_seed + 1),
        "regulatory_R1": DynParams(beta=1.0, steps=250, dt=0.04, seed=global_seed + 4),
        "epidemic": DynParams(beta=1.2, steps=300, dt=0.04, seed=global_seed + 2),
        "biochemical": DynParams(beta=1.2, steps=300, dt=0.04, seed=global_seed + 5),
        "mutualistic": DynParams(beta=0.2, steps=100, dt=0.02, seed=global_seed + 3),
        "regulatory_R2": DynParams(beta=0.2, steps=100, dt=0.02, seed=global_seed + 6),
    }
    samples = []
    for base_id in range(n_structures):
        graph_seed = global_seed + base_id * 10
        G = generate_base_graph(n_nodes, p_edge, graph_seed)
        A = to_csr(G)
        for label, (dyn_L1, dyn_L2) in enumerate(
            [
                ("population", "regulatory_R1"),
                ("epidemic", "biochemical"),
                ("mutualistic", "regulatory_R2"),
            ]
        ):
            sim_L1 = DYNAMICS[dyn_L1]
            sim_L2 = DYNAMICS[dyn_L2]

            p1 = dyn_params[dyn_L1]
            p2 = dyn_params[dyn_L2]

            x0_1 = random_initial_states(
                n_nodes, dyn_L1, seed=graph_seed + label * 100 + 1
            )
            x0_2 = random_initial_states(
                n_nodes, dyn_L2, seed=graph_seed + label * 100 + 2
            )

            p_layer1 = replace(p1, seed=graph_seed + label * 100 + 11)
            p_layer2 = replace(p2, seed=graph_seed + label * 100 + 12)

            xL1 = sim_L1(A, x0_1, p_layer1)
            xL2 = sim_L2(A, x0_2, p_layer2)

            samples.append(MultiplexGraph(A, A, xL1, xL2, base_id, label))

    return samples


def load_Syn4(g_n, n_nodes=100, p_edge=0.08, n_structures=30):
    GLOBAL_SEED = 2025
    samples = build_dataset(
        n_nodes=n_nodes,
        p_edge=p_edge,
        n_structures=n_structures,
        global_seed=GLOBAL_SEED,
    )
    s = samples[g_n]

    adj_mats = []
    adj_mats.append(s.A1)
    adj_mats.append(s.A2)

    label = torch.tensor([s.label]).float()

    feat_mats = [
        torch.FloatTensor(s.xL1).reshape(n_nodes, -1),
        torch.FloatTensor(s.xL2).reshape(n_nodes, -1),
    ]

    base_id = s.base_id

    return adj_mats, feat_mats, label, base_id
