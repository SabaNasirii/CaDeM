import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting sparse tensor to CSR format for more efficient processing.*",
)


def onehot_encoding(labels):
    labels = labels.reshape(-1, 1)
    encoder = OneHotEncoder()
    encoder.fit(labels)
    labels_onehot = encoder.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def preprocess_features_Syn1(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    return torch.sparse.FloatTensor(
        indices, torch.from_numpy(sparse_mx.data), torch.Size(sparse_mx.shape)
    )


def standardize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.0) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A


def dropping_features(x, p, seed) -> torch.Tensor:
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    rand = torch.rand(x.size(1), generator=gen, device=x.device)
    drop_mask = rand < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x
