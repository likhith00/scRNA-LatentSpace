from typing import Dict, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from sklearn.datasets import load_digits, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoaderFactory:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.batch_size = int(params["batch_size"])
        self.num_workers = int(params.get("num_workers", 0))

        # Optional per-dataset config
        # Example:
        # dataset_cfg:
        #   blobs: { n_samples: 20000, n_features: 100, centers: 10, cluster_std: 1.5, random_state: 0, test_size: 0.2 }
        self.dataset_cfg = params.get("dataset_cfg", {})

    # -------------------------
    # Image datasets (torchvision)
    # -------------------------

    def build_mnist_loaders(self) -> Tuple[DataLoader, DataLoader]:
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_ds = datasets.MNIST(
            root="./data", train=True, download=True,
            transform=mnist_transform
        )
        valid_ds = datasets.MNIST(
            root="./data", train=False, download=True,
            transform=mnist_transform
        )
        return self._wrap_loaders(train_ds, valid_ds)

    def build_fashion_mnist_loaders(self) -> Tuple[DataLoader, DataLoader]:
        fashion_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])

        train_ds = datasets.FashionMNIST(
            root="./data", train=True, download=True,
            transform=fashion_transform
        )
        valid_ds = datasets.FashionMNIST(
            root="./data", train=False, download=True,
            transform=fashion_transform
        )
        return self._wrap_loaders(train_ds, valid_ds)

    def build_emnist_balanced_loaders(self) -> Tuple[DataLoader, DataLoader]:
        # EMNIST images are also 28x28 grayscale.
        # Using MNIST-like normalization is usually fine for a baseline.
        emnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_ds = datasets.EMNIST(
            root="./data",
            split="balanced",
            train=True,
            download=True,
            transform=emnist_transform,
        )
        valid_ds = datasets.EMNIST(
            root="./data",
            split="balanced",
            train=False,
            download=True,
            transform=emnist_transform,
        )
        return self._wrap_loaders(train_ds, valid_ds)

    # -------------------------
    # Tabular datasets (sklearn)
    # -------------------------

    def build_digits_loaders(self) -> Tuple[DataLoader, DataLoader]:
        data = load_digits()
        X = data.data.astype("float32")
        y = data.target.astype("int64")

        X = StandardScaler().fit_transform(X).astype("float32")

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        valid_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
        return self._wrap_loaders(train_ds, valid_ds)

    def build_blobs_loaders(self) -> Tuple[DataLoader, DataLoader]:
        cfg = self.dataset_cfg.get("blobs", {})

        n_samples = int(cfg.get("n_samples", 20000))
        n_features = int(cfg.get("n_features", 100))
        centers = int(cfg.get("centers", 10))
        cluster_std = float(cfg.get("cluster_std", 1.5))
        random_state = int(cfg.get("random_state", 0))
        test_size = float(cfg.get("test_size", 0.2))

        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            random_state=random_state,
        )
        X = X.astype("float32")
        y = y.astype("int64")

        # Standardize so reconstruction + distance metrics behave sensibly
        X = StandardScaler().fit_transform(X).astype("float32")

        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if centers > 1 else None,
        )

        train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        valid_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
        return self._wrap_loaders(train_ds, valid_ds)

    # -------------------------
    # Utilities
    # -------------------------

    def _wrap_loaders(self, train_ds, valid_ds) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, valid_loader

    def build(self, dataset: str) -> Tuple[DataLoader, DataLoader]:
        dataset = (dataset or "").lower().strip()
        if dataset == "mnist":
            return self.build_mnist_loaders()
        if dataset in ("fashion_mnist", "fashion", "fmnist"):
            return self.build_fashion_mnist_loaders()
        if dataset in ("emnist_balanced", "emnist-balanced", "emnist"):
            return self.build_emnist_balanced_loaders()
        if dataset == "digits":
            return self.build_digits_loaders()
        if dataset == "blobs":
            return self.build_blobs_loaders()

        raise ValueError(
            f"Unknown dataset: {dataset}. "
            "Supported: mnist, fashion_mnist, emnist_balanced, digits, blobs"
        )
