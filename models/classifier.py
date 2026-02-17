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


class classifier(nn.Module):
    def __init__(self, ft_in, nb_classes, seed):
        super(classifier, self).__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(ft_in, nb_classes)
        self.weights_init()

    def weights_init(self):
        torch.nn.init.xavier_uniform_(self.fc.weight.data)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.0)

    def forward(self, seq):
        out = self.fc(seq)
        return out
