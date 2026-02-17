import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting sparse tensor to CSR format for more efficient processing.*",
)


class NodeMHAttentionCombiner(nn.Module):
    def __init__(self, embed_dim, num_heads, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.mha.in_proj_weight)
        if self.mha.in_proj_bias is not None:
            self.mha.in_proj_bias.data.fill_(0.0)

        torch.nn.init.xavier_uniform_(self.mha.out_proj.weight)
        if self.mha.out_proj.bias is not None:
            self.mha.out_proj.bias.data.fill_(0.0)

        torch.nn.init.xavier_uniform_(self.query.data)

    def forward(self, stacked):
        seq = stacked.transpose(0, 1)
        q = self.query.expand(-1, seq.size(1), -1)
        out, attn_weights = self.mha(q, seq, seq)
        out = out.squeeze(0)
        return out
