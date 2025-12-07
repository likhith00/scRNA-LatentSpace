# scRNA-LatentSpace
A comprehensive analysis of autoencoder architectures for single-cell RNA-seq data, focusing on the relationship between latent space dimensionality and biological class structure. The project aims to derive a principled rule for choosing latent dimensionality.

# How to execute

1. `python src/train.py --params "params.yaml" --dataset "mnist"`
2. `python src/eval.py --params "params.yaml" --dataset "mnist" --run-dir outputs/run_2dc5be4`
3. `python src/visualize.py --run-dir outputs/run_4620917 --umap`
