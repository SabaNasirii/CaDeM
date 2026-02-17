import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting sparse tensor to CSR format for more efficient processing.*",
)
import matplotlib.pyplot as plt


def plot_ari_nmi_vs_lambda(
    lambdas,
    common_mu,
    common_sd,
    priv0_mu,
    priv0_sd,
    priv1_mu,
    priv1_sd,
    priv2_mu,
    priv2_sd,
    title,
):

    plt.rcParams.update(
        {
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )

    colors = {
        "common": "#0072B2",
        "priv1": "#D55E00",
        "priv2": "#009E73",
        "priv3": "#CC79A7",
    }

    plt.figure(figsize=(9, 5))

    plt.errorbar(
        lambdas,
        common_mu,
        yerr=common_sd,
        fmt="-o",
        capsize=3,
        lw=2.0,
        ms=6,
        color=colors["common"],
        label=r"Common embeddings clustered on $c$",
    )

    plt.errorbar(
        lambdas,
        priv0_mu,
        yerr=priv0_sd,
        fmt="-o",
        capsize=3,
        lw=2.0,
        ms=6,
        color=colors["priv1"],
        label=r"Private embeddings (1st layer) clustered on $s_1$",
    )

    plt.errorbar(
        lambdas,
        priv1_mu,
        yerr=priv1_sd,
        fmt="-o",
        capsize=3,
        lw=2.0,
        ms=6,
        color=colors["priv2"],
        label=r"Private embeddings (2nd layer) clustered on $s_2$",
    )

    plt.errorbar(
        lambdas,
        priv2_mu,
        yerr=priv2_sd,
        fmt="-o",
        capsize=3,
        lw=2.0,
        ms=6,
        color=colors["priv3"],
        label=r"Private embeddings (3rd layer) clustered on $s_3$",
    )

    plt.xlabel(r"trade-off parameter $\lambda$", fontweight="bold")
    plt.ylabel(rf"{title} score (mean $\pm$ std over 50 runs)", fontweight="bold")
    plt.title(rf"{title} vs. trade-off parameter $\lambda$", fontweight="bold")

    plt.legend(frameon=False)
    plt.legend(frameon=False, prop={"weight": "normal"})

    plt.tight_layout()
    
    out_path = os.path.join('.', f"{title}_vs_lambda.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved in {out_path}")
    plt.show(block=True)
    