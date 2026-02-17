# Causality-Driven Disentangled Representation Learning in Multiplex Graphs

Learning representations from multiplex graphs, i.e., multi-layer networks where nodes interact through multiple relation types, is challenging due to the entanglement of shared (common) and layer-specific (private) information, which limits generalization and interpretability. In this work, we introduce a causal inference–based framework that disentangles common and private components in a self-supervised manner. **CaDeM** jointly (i) aligns shared embeddings across layers, (ii) enforces private embeddings to capture layer-specific signals, and (iii) applies backdoor adjustment to ensure that the common embeddings capture only global information while being separated from the private representations. Experiments on both synthetic and real-world datasets demonstrate consistent improvements over existing baselines, highlighting the effectiveness of our approach for robust and interpretable multiplex graph representation learning.

---

# Repository Structure
## Project Structure

```
CaDeM/
│
├── main.py
├── config.yaml
├── requirements.txt
│
├── train/
│   ├── train_network.py
│   └── CaDeM.py
│
├── evaluate/
│   ├── stratified_nested_CV.py
│   ├── stratified_grouped_nested_CV.py
│   └── get_ARI_NMI.py
│
├── visualize/
│   └── plot.py
│
├── load_data/
│   ├── load_real_world_datasets.py
│   ├── load_Syn_1.py
│   ├── load_Syn_2.py
│   ├── load_Syn_3.py
│   └── load_Syn_4.py
│
├── models/
│   ├── combiner.py
│   ├── classifier.py
│   ├── causal_heads.py
│   └── encoder_model.py
│
├── process_data/
│   └── data_processing.py
│
└── data/
    └── (dataset files)
```

# Environment Setup

We recommend using Conda with CUDA-enabled PyTorch.

## Create and activate environment

```bash
conda create -n CaDeM python=3.12.7 -y
conda activate CaDeM
```
## Install PyTorch

```bash
conda install pytorch==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
```
## Upgrade pip
```bash
python -m pip install --upgrade pip setuptools wheel
```
## Install PyTorch Geometric
```bash
pip install torch-geometric==2.6.1 torch_cluster==1.6.3+pt25cu124 torch_scatter==2.1.2+pt25cu124 torch_sparse==0.6.18+pt25cu124 torch_spline_conv==1.2.2+pt25cu124 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```
## Install remaining dependencies
```bash
pip install -r requirements.txt
```

# Running the Code
From the root directory:
```bash
python main.py --config_path "./config.yaml" --dataset "ACM"
```

## Arguments

##### `--config_path`
Path to the configuration file.

If you are in the root directory, use:

```bash
./config.yaml
```

##### `--dataset`
Name of the dataset to run experiments on.

Available datasets: ACM - DBLP - IMDB - freebase - Syn1 - Syn2 - Syn3 - Syn4

## Output
After running the code, the evaluation results will be displayed in the terminal. For certain experiments, performance plots are automatically generated. In addition, the framework outputs the learned common and private embeddings, which can be used for further analysis or downstream tasks.


# Changing Hyperparameters

All hyperparameters are defined in config.yaml. To modify training or evaluation settings, directly edit the corresponding dataset block.

### Some General Parameters

| Parameter     | Description |
|--------------|------------|
| `num_iters`  | Number of epochs |
| `patience`   | Early stopping patience |
| `gpu_num`    | GPU index (use `-1` for CPU) |
| `sparse`     | Whether adjacency matrices are treated as sparse |
| `hid_units`  | Hidden layer dimensionality of GCNs|
| `c_dim`   | Dimension of common embeddings |
| `p_dim`   | Dimension of private embeddings |
| `lr_causal` | Learning rate for causal heads |
| `lr_ED` | Learning rate for the encoder|
| `weight_decay_causal` | Weight decay for causal heads |
| `weight_decay_ED` | Weight decay for the encoder |
| `self_sup_coeff` | self-supervised loss weight|
| `causal_coeff` | causal loss weight |
| `matching_coeff` | matching loss weight |
| `feature_drop` | Feature dropout rate |
| `n_augment` | Number of graph augmentations per layer |
| `noise_std` | Noise level used in graph augmentation |
| `n_keep_nodes` | Proportion of nodes sampled in graph augmentation |
| `epochs_evaluation` | Number of epochs for evaluation classifier and combiner |
| `lr_evaluation` | Learning rate for evaluation classifier and combiner |


