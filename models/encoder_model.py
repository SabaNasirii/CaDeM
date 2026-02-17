import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from torch_geometric.nn.models import GCN
import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting sparse tensor to CSR format for more efficient processing.*",
)


class Encoder(nn.Module):
    def __init__(self, args, seed):
        super().__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.GNN_common = GCN(
            in_channels=args["ft_size"],
            hidden_channels=args["hid_units"],
            num_layers=1,
            dropout=0.0,
            act="relu",
            add_self_loops=False,
            bias=False,
        )

        self.GNN_private = GCN(
            in_channels=args["ft_size"],
            hidden_channels=args["hid_units"],
            num_layers=1,
            dropout=0.0,
            act="relu",
            add_self_loops=False,
            bias=False,
        )

        self.C = nn.Linear(args["hid_units"], args["c_dim"], bias=True)
        self.P = nn.Linear(args["hid_units"], args["p_dim"], bias=True)

        nn.init.xavier_uniform_(self.C.weight)
        nn.init.zeros_(self.C.bias)
        nn.init.xavier_uniform_(self.P.weight)
        nn.init.zeros_(self.P.bias)

    def forward(self, x, adj):
        common1 = self.GNN_common(x, adj)
        private1 = self.GNN_private(x, adj)
        common = self.C(private1)
        private = self.P(common1)
        return common, private


def update_S(model, features, adj_list, device, args):
    model.eval()
    commons = []
    with torch.no_grad():
        common, _ = model.encode(features, adj_list)
        commons.append(torch.cat(common, 1))
        commons = torch.cat(commons, 0)
        commons = commons - torch.mean(commons, 0, True)
        accumualte = []
        for i in range(args["num_view"]):
            accumualte.append(commons[:, i * args["c_dim"] : (i + 1) * args["c_dim"]])

        commons = torch.stack(accumualte, dim=2)
        U, _, V = torch.svd(torch.sum(commons, dim=2).cpu())
        S = torch.mm(U.to(device), V.to(device).t())
        S = S * (commons.shape[0]) ** 0.5
    return S


class Encoder_Module(nn.Module):
    def __init__(self, args, seed):
        super().__init__()
        self.args = args
        num_view = args["num_view"]
        self.encoder = nn.ModuleList()
        for i in range(num_view):
            enc_seed = seed + 2 * i
            self.encoder.append(Encoder(args, seed=enc_seed))

    def encode(self, x, adj_list):
        # get initaial node features and returns common and private parts
        common, private = [], []
        for i, enc in enumerate(self.encoder):
            c, p = enc(x[i], adj_list[i])
            common.append(c)
            private.append(p)
        return common, private

    def forward(self, x, adj_list):
        common, private = self.encode(x, adj_list)
        return common, private

    def embed(self, x, adj_list):
        common, private = [], []
        for i, enc in enumerate(self.encoder):
            c, p = enc(x[i], adj_list[i])
            common.append(c.detach())
            private.append(p.detach())
        return common, private
