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
import pickle
from scipy.io import loadmat
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from process_data.data_processing import *


def load_ACM(path):
    data = loadmat(path)
    labels = data["label"]
    adj1 = data["PLP"]
    adj2 = data["PAP"]

    adj_mats = []
    adj_mats.append(csr_matrix(adj1))
    adj_mats.append(csr_matrix(adj2))

    feat_mats = data["feature"]
    feat_mats = torch.FloatTensor(preprocess_features(feat_mats))

    labels = torch.tensor(labels).float()

    return adj_mats, feat_mats, labels


def load_IMDB(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    labels = data["label"]
    adj1 = data["MDM"]
    adj2 = data["MAM"]

    adj_mats = []
    adj_mats.append(csr_matrix(adj1))
    adj_mats.append(csr_matrix(adj2))

    feat_mats = data["feature"]
    feat_mats = torch.FloatTensor(preprocess_features(feat_mats))

    labels = torch.tensor(labels).float()

    return adj_mats, feat_mats, labels


def load_freebase(path):
    type_num = 3492
    label = np.load(path + "labels.npy").astype("int32")
    label = onehot_encoding(label)
    feat_m = np.eye(type_num)
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    label = torch.FloatTensor(label)
    feat_m = torch.FloatTensor(preprocess_features(feat_m))
    adj_list = [mam, mdm, mwm]

    return adj_list, feat_m, label


def load_DBLP(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    labels = data["label"]
    adj1 = data["PAP"]
    adj2 = data["PPrefP"]
    adj3 = data["PATAP"]

    adj_mats = []
    adj_mats.append(csr_matrix(adj1))
    adj_mats.append(csr_matrix(adj2))
    adj_mats.append(csr_matrix(adj3))

    feat_mats = data["feature"]
    feat_mats = torch.FloatTensor(preprocess_features(feat_mats))

    labels = torch.tensor(labels).float()

    return adj_mats, feat_mats, labels
