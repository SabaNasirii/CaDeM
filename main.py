import os


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting sparse tensor to CSR format for more efficient processing.*",
)
warnings.filterwarnings(
    "ignore", message="KMeans is known to have a memory leak on Windows with MKL"
)
import gc
import yaml
from copy import deepcopy
from train.CaDeM import *
from evaluate.stratified_nested_CV import *
from evaluate.get_ARI_NMI import *
from visualize.plot import *
from evaluate.stratified_grouped_nested_CV import *
import argparse


def run_CaDem(config_path, dataset_name):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    datasets = cfg["datasets"]
    if dataset_name not in datasets:
        available = ", ".join(datasets.keys())
        raise KeyError(f"Dataset '{dataset_name}' not found. Available: {available}")

    args = deepcopy(datasets[dataset_name])

    if dataset_name == "ACM":
        gc.collect()
        torch.cuda.empty_cache()
        embedder = CaDeM(args)
        common_learnt, private_learnt, labels = embedder.training()
        evaluate_stratified_nested_CV(
            common_learnt,
            private_learnt,
            labels,
            epoch=args["epochs_evaluation"],
            lr=args["lr_evaluation"],
            seed=args["seed_test"],
        )
        return {
            "common embeddings": common_learnt,
            "private embeddings": private_learnt,
            "labels": labels,
        }

    elif dataset_name == "IMDB":
        gc.collect()
        torch.cuda.empty_cache()
        embedder = CaDeM(args)
        common_learnt, private_learnt, labels = embedder.training()
        evaluate_stratified_nested_CV(
            common_learnt,
            private_learnt,
            labels,
            epoch=args["epochs_evaluation"],
            lr=args["lr_evaluation"],
            seed=args["seed_test"],
        )
        return {
            "common embeddings": common_learnt,
            "private embeddings": private_learnt,
            "labels": labels,
        }

    elif dataset_name == "freebase":
        gc.collect()
        torch.cuda.empty_cache()
        embedder = CaDeM(args)
        common_learnt, private_learnt, labels = embedder.training()
        evaluate_stratified_nested_CV(
            common_learnt,
            private_learnt,
            labels,
            epoch=args["epochs_evaluation"],
            lr=args["lr_evaluation"],
            seed=args["seed_test"],
        )
        return {
            "common embeddings": common_learnt,
            "private embeddings": private_learnt,
            "labels": labels,
        }

    elif dataset_name == "DBLP":
        gc.collect()
        torch.cuda.empty_cache()
        embedder = CaDeM(args)
        common_learnt, private_learnt, labels = embedder.training()
        evaluate_stratified_nested_CV(
            common_learnt,
            private_learnt,
            labels,
            epoch=args["epochs_evaluation"],
            lr=args["lr_evaluation"],
            seed=args["seed_test"],
        )
        return {
            "common embeddings": common_learnt,
            "private embeddings": private_learnt,
            "labels": labels,
        }

    elif dataset_name == "Syn1":
        embedder = CaDeM(args, label_probs=args["label_probs"])
        common_learnt, private_learnt, labels_views, final_labels = embedder.training()
        evaluate_stratified_nested_CV(
            common_learnt,
            private_learnt,
            final_labels,
            epoch=args["epochs_evaluation"],
            lr=args["lr_evaluation"],
            seed=args["seed_test"],
        )
        return {
            "common embeddings": common_learnt,
            "private embeddings": private_learnt,
            "final labels": final_labels,
        }

    elif dataset_name == "Syn2":
        lambdas = np.linspace(0, 1, 11)
        common_ari_mu, common_ari_sd = [], []
        priv0_ari_mu, priv0_ari_sd = [], []
        priv1_ari_mu, priv1_ari_sd = [], []
        priv2_ari_mu, priv2_ari_sd = [], []

        common_nmi_mu, common_nmi_sd = [], []
        priv0_nmi_mu, priv0_nmi_sd = [], []
        priv1_nmi_mu, priv1_nmi_sd = [], []
        priv2_nmi_mu, priv2_nmi_sd = [], []

        for lam in tqdm(lambdas):
            gc.collect()
            torch.cuda.empty_cache()

            args["lambda"] = lam
            embedder = CaDeM(args)
            common_learnt, private_learnt, common_labels, private_labels = (
                embedder.training()
            )

            C = common_learnt.detach().cpu().numpy()
            P0 = private_learnt[0].detach().cpu().numpy()
            P1 = private_learnt[1].detach().cpu().numpy()
            P2 = private_learnt[2].detach().cpu().numpy()

            yC = common_labels.argmax(dim=1).detach().cpu().numpy()
            yP0 = private_labels[0].argmax(dim=1).detach().cpu().numpy()
            yP1 = private_labels[1].argmax(dim=1).detach().cpu().numpy()
            yP2 = private_labels[2].argmax(dim=1).detach().cpu().numpy()

            # Run KMeans 50 times
            mu, sd, nmi_mu, nmi_sd = kmeans_ari_nmi_stats(
                C, yC, K=args["n_communities"], runs=50
            )
            common_ari_mu.append(mu)
            common_ari_sd.append(sd)
            common_nmi_mu.append(nmi_mu)
            common_nmi_sd.append(nmi_sd)

            mu, sd, nmi_mu, nmi_sd = kmeans_ari_nmi_stats(
                P0, yP0, K=args["n_communities"], runs=50
            )
            priv0_ari_mu.append(mu)
            priv0_ari_sd.append(sd)
            priv0_nmi_mu.append(nmi_mu)
            priv0_nmi_sd.append(nmi_sd)

            mu, sd, nmi_mu, nmi_sd = kmeans_ari_nmi_stats(
                P1, yP1, K=args["n_communities"], runs=50
            )
            priv1_ari_mu.append(mu)
            priv1_ari_sd.append(sd)
            priv1_nmi_mu.append(nmi_mu)
            priv1_nmi_sd.append(nmi_sd)

            mu, sd, nmi_mu, nmi_sd = kmeans_ari_nmi_stats(
                P2, yP2, K=args["n_communities"], runs=50
            )
            priv2_ari_mu.append(mu)
            priv2_ari_sd.append(sd)
            priv2_nmi_mu.append(nmi_mu)
            priv2_nmi_sd.append(nmi_sd)

        plot_ari_nmi_vs_lambda(
            lambdas,
            common_ari_mu,
            common_ari_sd,
            priv0_ari_mu,
            priv0_ari_sd,
            priv1_ari_mu,
            priv1_ari_sd,
            priv2_ari_mu,
            priv2_ari_sd,
            "ARI",
        )
        plot_ari_nmi_vs_lambda(
            lambdas,
            common_nmi_mu,
            common_nmi_sd,
            priv0_nmi_mu,
            priv0_nmi_sd,
            priv1_nmi_mu,
            priv1_nmi_sd,
            priv2_nmi_mu,
            priv2_nmi_sd,
            "NMI",
        )

        return {
            "mean ARI common": common_ari_mu,
            "mean ARI private 1": priv0_ari_mu,
            "mean ARI private 2": priv1_ari_mu,
            "mean ARI private 3": priv2_ari_mu,
            "mean NMI common": common_nmi_mu,
            "mean NMI private 1": priv0_nmi_mu,
            "mean NMI private 2": priv1_nmi_mu,
            "mean NMI private 3": priv2_nmi_mu,
        }

    elif dataset_name == "Syn3":
        gc.collect()
        torch.cuda.empty_cache()
        embedder = CaDeM(args)
        common_learnt, private_learnt, labels_1, labels_2 = embedder.training()

        common_mean_ari_1, common_mean_nmi_1 = kmeans_ari_nmi_mean(
            common_learnt.detach().cpu().numpy(),
            (labels_1.argmax(dim=1).detach().cpu().numpy() == 3).astype(int),
            K=2,
            n_runs=50,
        )
        print(
            f"ARI common embeddings on labels of layer 1: "
            f"{common_mean_ari_1:.4f}, "
            f"NMI common embeddings on labels of layer 1: {common_mean_nmi_1:.4f}"
        )
        common_mean_ari_2, common_mean_nmi_2 = kmeans_ari_nmi_mean(
            common_learnt.detach().cpu().numpy(),
            (labels_2.argmax(dim=1).detach().cpu().numpy() == 3).astype(int),
            K=2,
            n_runs=50,
        )
        print(
            f"ARI common embeddings on labels of layer 2: "
            f"{common_mean_ari_2:.4f}, "
            f"NMI common embeddings on labels of layer 2: {common_mean_nmi_2:.4f}"
        )

        priv1_mean_ari_1, priv1_mean_nmi_1 = kmeans_ari_nmi_mean(
            private_learnt[0].detach().cpu().numpy(),
            (labels_1.argmax(dim=1).detach().cpu().numpy() == 3).astype(int),
            K=2,
            n_runs=50,
        )
        print(
            f"ARI private embeddings (1st layer) on labels of layer 1: "
            f"{priv1_mean_ari_1:.4f}, "
            f"NMI private embeddings (1st layer) on labels of layer 1: {priv1_mean_nmi_1:.4f}"
        )
        priv1_mean_ari_2, priv1_mean_nmi_2 = kmeans_ari_nmi_mean(
            private_learnt[0].detach().cpu().numpy(),
            (labels_2.argmax(dim=1).detach().cpu().numpy() == 3).astype(int),
            K=2,
            n_runs=50,
        )
        print(
            f"ARI private embeddings (1st layer) on labels of layer 2: "
            f"{priv1_mean_ari_2:.4f}, "
            f"NMI private embeddings (1st layer) on labels of layer 2: {priv1_mean_nmi_2:.4f}"
        )

        priv2_mean_ari_1, priv2_mean_nmi_1 = kmeans_ari_nmi_mean(
            private_learnt[1].detach().cpu().numpy(),
            (labels_1.argmax(dim=1).detach().cpu().numpy() == 3).astype(int),
            K=2,
            n_runs=50,
        )
        print(
            f"ARI private embeddings (2nd layer) on labels of layer 1: "
            f"{priv2_mean_ari_1:.4f}, "
            f"NMI private embeddings (2nd layer) on labels of layer 1: {priv2_mean_nmi_1:.4f}"
        )

        priv2_mean_ari_2, priv2_mean_nmi_2 = kmeans_ari_nmi_mean(
            private_learnt[1].detach().cpu().numpy(),
            (labels_2.argmax(dim=1).detach().cpu().numpy() == 3).astype(int),
            K=2,
            n_runs=50,
        )
        print(
            f"ARI private embeddings (2nd layer) on labels of layer 2: "
            f"{priv2_mean_ari_2:.4f}, "
            f"NMI private embeddings (2nd layer) on labels of layer 2: {priv2_mean_nmi_2:.4f}"
        )

        return {
            "common embeddings": common_learnt,
            "private embeddings": private_learnt,
            "labels of layer 1": labels_1,
            "labels of layer 2": labels_2,
        }

    elif dataset_name == "Syn4":
        commons = []
        privates1 = []
        privates2 = []
        labels = []
        base_ids = []

        for i in tqdm(range(90)):
            gc.collect()
            torch.cuda.empty_cache()
            args["g_n"] = i
            embedder = CaDeM(args)
            common_learnt, private_learnt, label, base_id = embedder.training()

            commons.append(common_learnt)
            privates1.append(private_learnt[0])
            privates2.append(private_learnt[1])

            labels.append(label)
            base_ids.append(base_id)

        pooled_common_embeddings = []
        for i in commons:
            pooled_common_embeddings.append(i.mean(0))
        pooled_common_embeddings = torch.stack(pooled_common_embeddings)

        pooled_private1_embeddings = []
        for i in privates1:
            pooled_private1_embeddings.append(i.mean(0))
        pooled_private1_embeddings = torch.stack(pooled_private1_embeddings)

        pooled_private2_embeddings = []
        for i in privates2:
            pooled_private2_embeddings.append(i.mean(0))
        pooled_private2_embeddings = torch.stack(pooled_private2_embeddings)

        New_labels = onehot_encoding(torch.stack(labels).cpu().detach().numpy())

        print(
            "Stratified Grouped Nested CV for Three-Class Dynamic Prediction with Combined Embeddings"
        )
        evaluate_stratified_grouped_nested_CV(
            pooled_common_embeddings,
            [pooled_private1_embeddings, pooled_private2_embeddings],
            torch.tensor(New_labels),
            np.array(base_ids),
            args["epochs_evaluation"],
            0.01,
            args["seed_test"],
        )

        print(
            "Stratified Grouped Nested CV for Three-Class Dynamic Prediction with Common Embeddings"
        )
        evaluate_stratified_grouped_nested_CV_single_embedding(
            pooled_common_embeddings,
            torch.tensor(New_labels),
            np.array(base_ids),
            args["epochs_evaluation"],
            args["lr_evaluation"],
            args["seed_test"],
        )

        print(
            "Stratified Grouped Nested CV for Three-Class Dynamic Prediction with Private Embeddings (1st layer)"
        )
        evaluate_stratified_grouped_nested_CV_single_embedding(
            pooled_private1_embeddings,
            torch.tensor(New_labels),
            np.array(base_ids),
            args["epochs_evaluation"],
            args["lr_evaluation"],
            args["seed_test"],
        )

        print(
            "Stratified Grouped Nested CV for Three-Class Dynamic Prediction with Private Embeddings (2nd layer)"
        )
        evaluate_stratified_grouped_nested_CV_single_embedding(
            pooled_private2_embeddings,
            torch.tensor(New_labels),
            np.array(base_ids),
            args["epochs_evaluation"],
            args["lr_evaluation"],
            args["seed_test"],
        )

        labels_p1 = np.argmax(New_labels, 1) * 2
        labels_p2 = np.argmax(New_labels, 1) * 2 + 1
        private_embs_6class = torch.cat(
            [pooled_private1_embeddings, pooled_private2_embeddings], dim=0
        )
        labels_6class = np.concatenate([labels_p1, labels_p2], 0)
        labels_6class = onehot_encoding(labels_6class)
        base_ids_6class = np.concatenate([base_ids, base_ids], 0)

        print(
            "Stratified Grouped Nested CV for Six-Subclass Dynamic Prediction with Private Embeddings"
        )
        evaluate_stratified_grouped_nested_CV_single_embedding(
            private_embs_6class,
            torch.tensor(labels_6class),
            np.array(base_ids_6class),
            args["epochs_evaluation"],
            args["lr_evaluation"],
            args["seed_test"],
        )

        return {
            "pooled common embeddings": pooled_common_embeddings,
            "pooled private embeddings 1": pooled_private1_embeddings,
            "pooled private embeddings 2": pooled_private2_embeddings,
            "labels": New_labels,
            "base graph ids": base_ids,
        }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run CaDeM experiments")

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to config.yaml",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (must match key in config.yaml)",
    )

    args_cli = parser.parse_args()

    results = run_CaDem(
        config_path=args_cli.config_path,
        dataset_name=args_cli.dataset,
    )
