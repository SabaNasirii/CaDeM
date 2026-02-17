import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
import numpy as np
import random
from tqdm import tqdm
import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting sparse tensor to CSR format for more efficient processing.*",
)

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from load_data.load_real_world_datasets import *
from load_data.load_Syn_1 import *
from load_data.load_Syn_2 import *
from load_data.load_Syn_3 import *
from load_data.load_Syn_4 import *
from models.causal_heads import *
from models.encoder_model import *
from train.train_network import *


class Initializer:
    def __init__(self, args, label_probs=None):

        seed = args["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if args["gpu_num"] == -1:
            args["device"] = "cpu"
        else:
            args["device"] = "cuda"

        if args["dataset"] == "ACM":
            adj_list, features, labels = load_ACM(args["path"])
        elif args["dataset"] == "IMDB":
            adj_list, features, labels = load_IMDB(args["path"])
        elif args["dataset"] == "freebase":
            adj_list, features, labels = load_freebase(args["path"])
        elif args["dataset"] == "DBLP":
            adj_list, features, labels = load_DBLP(args["path"])
        elif args["dataset"] == "Syn1":
            adj_list, features, labels, self.labels_views = load_Syn1(
                n_nodes=args["n_nodes"],
                n_graphs=args["num_view"],
                n_communities=args["n_communities"],
                p_intra=args["p_intra"],
                p_inter=args["p_inter"],
                label_probs=label_probs,
            )
        elif args["dataset"] == "Syn2":
            adj_list, features, common_labels, private_labels = load_Syn2(
                n=args["n_node"],
                K=args["n_communities"],
                L=args["num_view"],
                pin=args["pin"],
                pout=args["pout"],
                pin_priv=args["pin_priv"],
                pout_priv=args["pout_priv"],
                lam=args["lambda"],
                seed=seed,
                portion=args["portion"],
            )
        elif args["dataset"] == "Syn3":
            adj_list, features, labels_1, labels_2 = load_Syn3(
                N=args["n_node"],
                k=args["n_neighbors"],
                m=args["n_communities"],
                alpha_h=args["alpha_h"],
                snr_common=args["snr_common"],
                snr_private=args["snr_private"],
            )
        elif args["dataset"] == "Syn4":
            adj_list, features, labels, base_id = load_Syn4(
                args["g_n"],
                n_nodes=args["n_node"],
                p_edge=args["p_edge"],
                n_structures=args["n_structures"],
            )

        adj_list = [
            sparse_mx_to_torch_sparse_tensor(adj).to_dense() for adj in adj_list
        ]
        adj_list = [standardize_graph(adj) for adj in adj_list]

        if args["sparse"]:
            adj_list = [adj.to_sparse() for adj in adj_list]

        args["nb_nodes"] = adj_list[0].shape[0]

        if args["dataset"] == "Syn2":
            args["nb_classes"] = common_labels.shape[1]
        elif args["dataset"] == "Syn3":
            args["nb_classes"] = labels_1.shape[1]
        elif args["dataset"] == "Syn4":
            args["nb_classes"] = 3
        else:
            args["nb_classes"] = labels.shape[1]

        if args["dataset"] == "Syn3" or args["dataset"] == "Syn4":
            args["ft_size"] = features[0].shape[1]
        else:
            args["ft_size"] = features.shape[1]

        self.adj_list = adj_list

        if args["dataset"] == "Syn3" or args["dataset"] == "Syn4":
            self.features = [torch.FloatTensor(f).to(args["device"]) for f in features]
        else:
            self.features = [
                torch.FloatTensor(features).to(args["device"])
                for _ in range(args["num_view"])
            ]

        if args["dataset"] == "Syn2":
            self.common_labels = torch.FloatTensor(common_labels).to(args["device"])
            self.private_labels = private_labels
        elif args["dataset"] == "Syn3":
            self.labels_1 = labels_1.to(args["device"])
            self.labels_2 = labels_2.to(args["device"])
        elif args["dataset"] == "Syn4":
            self.labels = labels.to(args["device"])
            self.base_id = base_id
        else:
            self.labels = torch.FloatTensor(labels).to(args["device"])

        self.args = args


class CaDeM(Initializer):
    def __init__(self, args, label_probs=None):
        super().__init__(args, label_probs)
        self.args = args

    def training(self):
        seed = self.args["seed"]
        device = self.args["device"]
        n_view = self.args["num_view"]

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        features = [f.to(device) for f in self.features]
        adj_list = [adj.to(device) for adj in self.adj_list]

        for i in range(n_view):
            features[i] = dropping_features(
                features[i], self.args["feature_drop"], seed + 10 + i
            )

        ED_model = Encoder_Module(self.args, seed=seed + 1000).to(device)
        causal_model = Causal_Networks(self.args, seed=seed + 2000).to(device)

        optimizer = torch.optim.Adam(
            [
                {
                    "params": causal_model.parameters(),
                    "lr": self.args["lr_causal"],
                    "weight_decay": self.args["weight_decay_causal"],
                },
                {
                    "params": ED_model.parameters(),
                    "lr": self.args["lr_ED"],
                    "weight_decay": self.args["weight_decay_ED"],
                },
            ]
        )

        best, waiting = float("inf"), 0

        for out_epoch in tqdm(
            range(1, self.args["num_iters"] + 1),
            disable=(self.args["dataset"] == "Syn4"),
        ):
            S = update_S(ED_model, features, adj_list, device, self.args)

            for in_epoch in range(self.args["inner_epochs"]):
                iter_seed = seed + out_epoch * 100 + in_epoch
                random.seed(iter_seed)
                np.random.seed(iter_seed)
                torch.manual_seed(iter_seed)
                torch.cuda.manual_seed_all(iter_seed)

                (loss, common, private) = train_models(
                    ED_model,
                    S,
                    features,
                    adj_list,
                    self.args,
                    optimizer,
                    device,
                    out_epoch * in_epoch,
                    causal_model,
                )

            with torch.no_grad():
                S = update_S(ED_model, features, adj_list, device, self.args)

            if loss < best:
                best, waiting = loss, 0
            else:
                waiting += 1
            if waiting >= self.args["patience"] and out_epoch > 100:
                print("Early stopping")
                break

        ED_model.eval()
        causal_model.eval()
        S_final = update_S(ED_model, features, adj_list, device, self.args)
        _, private_final = ED_model.embed(features, adj_list)

        if self.args["dataset"] == "Syn1":
            return (
                S_final,
                private_final,
                self.labels_views,
                self.labels,
            )
        elif self.args["dataset"] == "Syn2":
            return (S_final, private_final, self.common_labels, self.private_labels)
        elif self.args["dataset"] == "Syn3":
            return (S_final, private_final, self.labels_1, self.labels_2)
        elif self.args["dataset"] == "Syn4":
            return (S_final, private_final, self.labels, self.base_id)
        else:
            return (
                S_final,
                private_final,
                self.labels,
            )
