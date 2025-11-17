from typing import Dict, Tuple
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


class DataLoaderFactory:
    def __init__(self, params: Dict):
        self.batch_size = int(params["batch_size"])
        self.num_workers = int(params.get("num_workers", 0))

    def build_mnist_loaders(self) -> Tuple[DataLoader, DataLoader]:

        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        valid_ds = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return train_loader, valid_loader

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

        train_loader = DataLoader(train_ds, 
                                  batch_size=self.batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(valid_ds, 
                                  batch_size=self.batch_size,
                                  shuffle=False)
        return train_loader, valid_loader

    def build(self, dataset: str) -> Tuple[DataLoader, DataLoader]:
        if dataset == "mnist":
            return self.build_mnist_loaders()
        elif dataset == "digits":
            return self.build_digits_loaders()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
