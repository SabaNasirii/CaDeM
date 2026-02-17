import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting sparse tensor to CSR format for more efficient processing.*",
)
from scipy.io import loadmat

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from process_data.data_processing import *


class GraphDataset(Dataset):
    def __init__(self, data_label_list):
        self.data_label_list = data_label_list

    def __len__(self):
        return len(self.data_label_list)

    def __getitem__(self, idx):
        (data1, data2), label = self.data_label_list[idx]
        data_obj = Data()
        data_obj.x1 = data1.float()
        data_obj.x2 = data2.float()
        data_obj.y = torch.tensor([label], dtype=torch.float)
        return data_obj


def matching_loss(common, S, args):
    loss = torch.nn.MSELoss(reduction="sum")
    match_err = (
        loss(torch.cat(common, 1), S.repeat(1, args["num_view"])) / common[0].shape[0]
    )
    return match_err


def train_models(
    model, S, features, adj_list, args, optimizer, device, epoch, causal_model
):
    model.train()
    causal_model.train()

    common, private = model(features, adj_list)
    matching_error = matching_loss(common, S, args)

    mean, std_dev = 0.0, args["noise_std"]
    N_keep = int(args["n_keep_nodes"] * common[0].shape[0])
    n_view = args["num_view"]
    n_aug = args["n_augment"]
    base_seed = args["seed"]

    loss_causal = 0.0

    if n_aug != 0:
        graph_list = []
        for i in range(n_view):
            c_view = common[i]
            p_view = private[i]
            num_nodes = c_view.shape[0]

            for j in range(n_aug):
                rng_idx = np.random.default_rng(base_seed + 10_000 + i * 100 + j)
                idxs = rng_idx.choice(num_nodes, size=N_keep, replace=False)
                new_c = c_view[idxs]
                new_p = p_view[idxs]

                rng_c = np.random.default_rng(base_seed + 20_000 + i * 100 + j)
                noise_c = (
                    torch.from_numpy(
                        rng_c.normal(loc=mean, scale=std_dev, size=new_c.shape)
                    )
                    .to(device)
                    .float()
                )

                rng_p = np.random.default_rng(base_seed + 30_000 + i * 100 + j)
                noise_p = (
                    torch.from_numpy(
                        rng_p.normal(loc=mean, scale=std_dev, size=new_p.shape)
                    )
                    .to(device)
                    .float()
                )

                graph_list.append(((new_c + noise_c, new_p + noise_p), i))

        dl_gen = torch.Generator().manual_seed(base_seed + 40_000)
        graph_dataset = GraphDataset(graph_list)
        graph_loader = DataLoader(
            graph_dataset, batch_size=64, shuffle=True, generator=dl_gen, num_workers=0
        )

        for graphs in graph_loader:
            graphs = graphs.to(device)
            y = graphs.y.view(-1)
            one_hot = torch.tensor(
                onehot_encoding(y.cpu().numpy()), device=device, dtype=torch.float
            )
            p_log, cp_log = causal_model(graphs, n_nodes=N_keep)
            target = one_hot.argmax(dim=1)
            loss_causal += args["self_sup_coeff"] * F.nll_loss(p_log, target) + args[
                "causal_coeff"
            ] * F.nll_loss(cp_log, target)

    graph_list = []
    for i in range(n_view):
        c_view = common[i]
        p_view = private[i]

        rng_c = np.random.default_rng(base_seed + 50_000 + i)
        noise_c = torch.from_numpy(
            rng_c.normal(loc=mean, scale=std_dev, size=c_view.shape)
        ).to(device)

        rng_p = np.random.default_rng(base_seed + 60_000 + i)
        noise_p = torch.from_numpy(
            rng_p.normal(loc=mean, scale=std_dev, size=p_view.shape)
        ).to(device)

        graph_list.append(((c_view + noise_c, p_view + noise_p), i))

    graph_dataset = GraphDataset(graph_list)
    graph_loader = DataLoader(
        graph_dataset,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(base_seed + 70_000),
        num_workers=0,
    )

    for graphs in graph_loader:
        graphs = graphs.to(device)
        y = graphs.y.view(-1)
        one_hot = torch.tensor(
            onehot_encoding(y.cpu().numpy()), device=device, dtype=torch.float
        )
        p_log, cp_log = causal_model(graphs, n_nodes=common[0].shape[0])
        target = one_hot.argmax(dim=1)
        loss_causal += args["self_sup_coeff"] * F.nll_loss(p_log, target) + args[
            "causal_coeff"
        ] * F.nll_loss(cp_log, target)

    loss = args["matching_coeff"] * matching_error + loss_causal
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return (loss, common, private)
