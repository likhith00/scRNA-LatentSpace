from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoaderFactory:
    def __init__(self, params: Dict):
        self.batch_size = int(params["batch_size"])
        self.num_workers = int(params.get("num_workers", 0))

    def build_mnist_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        MNIST: 
        - ToTensor() skaliert von [0, 255] auf [0, 1]
        - Normalize zentriert / skaliert mit Standard-MNIST-Statistiken
        """

        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_ds = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=mnist_transform,
        )
        valid_ds = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=mnist_transform,
        )

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

    def build_digits_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        sklearn digits:
        - StandardScaler für Feature-Standardisierung
        - Konvertierung in float32 / int64 Tensors
        - optionales Reshaping kommentiert (falls du ein CNN nutzen willst)
        """

        data = load_digits()
        X = data.data.astype("float32")   
        y = data.target.astype("int64")

        X = StandardScaler().fit_transform(X).astype("float32")

        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        Xtr_t = torch.from_numpy(Xtr)
        Xte_t = torch.from_numpy(Xte)
        ytr_t = torch.from_numpy(ytr)
        yte_t = torch.from_numpy(yte)

        train_ds = TensorDataset(Xtr_t, ytr_t)
        valid_ds = TensorDataset(Xte_t, yte_t)

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
        if dataset == "mnist":
            return self.build_mnist_loaders()
        elif dataset == "digits":
            return self.build_digits_loaders()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
