import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch.nn import Linear
import torch.nn as nn
import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting sparse tensor to CSR format for more efficient processing.*",
)


class Causal_Networks(nn.Module):
    def __init__(self, args, seed):
        super().__init__()
        self.args = args
        self.global_pool = global_add_pool

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.np_rng = np.random.default_rng(seed)
        self.fc_p = Linear(args["p_dim"], args["num_view"])
        self.fc_cp = Linear(args["c_dim"], args["num_view"])

        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph, n_nodes: int = None):
        common = graph.x1
        private = graph.x2

        batch_size = common.shape[0] / n_nodes
        device = common.device

        batch_common = []
        for i in np.arange(batch_size):
            for j in np.arange(n_nodes):
                batch_common.append(i)
        batch_common = torch.tensor(batch_common).to(device).to(torch.int64)

        batch_private = []
        for i in np.arange(batch_size):
            for j in np.arange(n_nodes):
                batch_private.append(i)
        batch_private = torch.tensor(batch_private).to(device).to(torch.int64)

        x_common = self.global_pool(common, batch_common)
        x_private = self.global_pool(private, batch_private)

        x_private_logits = self.phi(self.fc_p, x_private)
        x_cp_logits = self.psi(x_common, x_private)

        return x_private_logits, x_cp_logits

    def phi(self, fc, x):
        x = fc(x)
        return F.log_softmax(x, dim=-1)

    def psi(self, xc, xp):
        num = xc.shape[0]
        perm = self.np_rng.permutation(num)
        idx = torch.from_numpy(perm).to(xc.device)
        x = xc[idx] + xp
        x = self.fc_cp(x)
        return F.log_softmax(x, dim=-1)
