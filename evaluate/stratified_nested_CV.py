import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
import numpy as np
import random
import torch.nn as nn
import warnings
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings(
    "ignore",
    message="Converting sparse tensor to CSR format for more efficient processing.*",
)

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.combiner import *
from models.classifier import *


@torch.no_grad()
def _eval_on_indices(combiner, classifier, stacked, y, idx):
    combiner.eval()
    classifier.eval()
    fused_all = combiner(stacked)
    logits = classifier(fused_all[idx])
    preds = logits.argmax(dim=1).cpu().numpy()
    true = y[idx]
    macro = f1_score(true, preds, average="macro")
    micro = f1_score(true, preds, average="micro")
    return macro, micro


def evaluate_stratified_nested_CV(
    commons, privates, labels, epoch, lr, seed, outer_splits=5, inner_splits=5
):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = (
        commons.device
        if hasattr(commons, "device")
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    commons = commons.to(device)
    privates = [p.to(device) for p in privates]
    labels = labels.to(device)

    y = labels.argmax(dim=1).cpu().numpy()
    stacked = torch.stack([commons] + privates, dim=1).to(device)

    outer_skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    inner_macro_means, inner_micro_means = [], []
    outer_macro_f1s, outer_micro_f1s = [], []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_skf.split(np.zeros(len(y)), y), start=1
    ):
        fold_seed = seed + outer_fold
        torch.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)
        np.random.seed(fold_seed)
        random.seed(fold_seed)

        inner_skf = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=fold_seed
        )
        y_outer_train = y[outer_train_idx]

        inner_macros, inner_micros = [], []

        for inner_train_rel, inner_val_rel in inner_skf.split(
            np.zeros(len(y_outer_train)), y_outer_train
        ):
            inner_train_idx = outer_train_idx[inner_train_rel]
            inner_val_idx = outer_train_idx[inner_val_rel]

            combiner = NodeMHAttentionCombiner(
                embed_dim=commons.size(1), num_heads=4, seed=fold_seed
            ).to(device)
            clssifier = classifier(commons.size(1), labels.size(1), seed=fold_seed).to(
                device
            )

            optimizer = torch.optim.Adam(
                list(combiner.parameters()) + list(clssifier.parameters()), lr=lr
            )
            criterion = nn.CrossEntropyLoss()

            for _ in range(epoch):
                combiner.train()
                clssifier.train()
                optimizer.zero_grad()
                fused = combiner(stacked)
                logits = clssifier(fused[inner_train_idx])
                loss = criterion(
                    logits, torch.tensor(y[inner_train_idx], device=device)
                )
                loss.backward()
                optimizer.step()

            macro, micro = _eval_on_indices(
                combiner, clssifier, stacked, y, inner_val_idx
            )
            inner_macros.append(macro)
            inner_micros.append(micro)

        inner_macro_mean = float(np.mean(inner_macros))
        inner_micro_mean = float(np.mean(inner_micros))
        inner_macro_means.append(inner_macro_mean)
        inner_micro_means.append(inner_micro_mean)

        combiner = NodeMHAttentionCombiner(
            embed_dim=commons.size(1), num_heads=4, seed=fold_seed
        ).to(device)
        clssifier = classifier(commons.size(1), labels.size(1), seed=fold_seed).to(
            device
        )

        optimizer = torch.optim.Adam(
            list(combiner.parameters()) + list(clssifier.parameters()), lr=lr
        )
        criterion = nn.CrossEntropyLoss()

        for _ in range(epoch):
            combiner.train()
            clssifier.train()
            optimizer.zero_grad()
            fused = combiner(stacked)
            logits = clssifier(fused[outer_train_idx])
            loss = criterion(logits, torch.tensor(y[outer_train_idx], device=device))
            loss.backward()
            optimizer.step()

        macro_test, micro_test = _eval_on_indices(
            combiner, clssifier, stacked, y, outer_test_idx
        )
        outer_macro_f1s.append(macro_test)
        outer_micro_f1s.append(micro_test)

    print(
        f"Nested CV – Outer Test Performance:"
        f" Macro-F1 = {np.mean(outer_macro_f1s):.4f} ± {np.std(outer_macro_f1s):.4f} |"
        f" Micro-F1 = {np.mean(outer_micro_f1s):.4f} ± {np.std(outer_micro_f1s):.4f}"
    )
