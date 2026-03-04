# Latent Space Dimensionality Analysis for Autoencoders

This repository contains the code and experiments for a systematic empirical study on latent space dimensionality selection in autoencoders.
The project investigates whether the number of classes in a dataset can be used as a reliable heuristic for choosing the latent dimension, and demonstrates that optimal latent dimensionality is dataset-specific and weakly related to label count.

The study evaluates autoencoders across multiple datasets using clustering, neighborhood preservation, and reconstruction-based metrics.

## Key Contributions

- Systematic coarse and fine sweeps over latent dimensionality

- Evaluation across multiple datasets with varying complexity and class counts

## Demonstrates that:

- Reconstruction loss decreases monotonically with latent dimension

- Representation quality (clustering) peaks at intermediate, dataset-specific dimensions

- Number of classes alone is not sufficient to select latent dimensionality

- Latent space redundancy analysis using correlation matrices and effective dimensionality

```
scRNA-LatentSpace/
│
├── src/
│   ├── train.py            # Train autoencoders for different latent dimensions
│   ├── eval.py             # Evaluate clustering, neighborhood, and redundancy metrics
│   ├── visualize.py        # Plot results and latent-space analyses
│   ├── dataloader.py       # Dataset loaders (MNIST, EMNIST, GTSRB, UCI HAR, etc.)
│   └── models/
│       ├── base.py         # Base autoencoder definitions
│       └── mlp.py          # MLP-based autoencoder implementation
├── bashscript_train.sh
├── bashscript_val.sh
├── bashscript_visualize.sh
├── params.yaml             # Experiment configuration
├── outputs/                # Saved models, metrics, and plots
├── data/                   # Automatically downloaded datasets
└── README.md
```

# Datasets Supported

- Digits (sklearn)

- MNIST

- Fashion-MNIST

- EMNIST (balanced)

- GTSRB (German Traffic Sign Recognition Benchmark)

Follow these steps in windows
# Installation
```
git clone https://github.com/likhith00/scRNA-LatentSpace.git
cd scRNA-LatentSpace
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```


# Running Experiments 
## Train autoencoders
```
python src/train.py --dataset digits
```

## Evaluate trained models
```
python src/eval.py --dataset digits --run-dir outputs/run_xxxxxxx
```

## Visualize results
```
python src/visualize.py --run-dir outputs/run_xxxxxxx --umap
```

# submitting slurm jobs
- edit bashscript_train.sh and select appropriate dataset and execute following command in terminal for training.
```sbatch bashscript_train.sh```
- to perform evaluation, edit bashscript_val.sh and set exact run id and execute the following command
```sbatch bashscript_val.sh```
- To visualize, set exact run id and execute following command
```sbatch bashscript_visualize.sh```



# Evaluation metrics
The following metrics are used to evaluate latent representations:

**Clustering:** ARI, NMI

**Neighborhood preservation:** Trustworthiness, Continuity

**Cluster geometry:** Silhouette, Davies–Bouldin, Calinski–Harabasz

# Main Findings

- Optimal latent dimensionality is not directly related to the number of classes.

- Datasets with the same number of labels often require very different latent sizes.

- Larger latent spaces improve reconstruction but introduce redundancy that harms clustering.

- Latent dimensionality should be selected based on downstream representation quality, not reconstruction loss alone.
