from typing import Dict, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import os
import urllib.request
import zipfile
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
    
    def build_uci_har_loaders(self):
        """
        UCI HAR:
        - downloads and extracts the dataset under dataset_cfg.uci_har.root
        - robustly finds the extracted folder that contains train/X_train.txt
        - loads X_train/y_train and X_test/y_test
        - StandardScaler fit on train, applied to test
        - labels converted from 1..6 to 0..5
        """
        import os
        import urllib.request
        import zipfile
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler

        cfg = self.params.get("dataset_cfg", {}).get("uci_har", {})
        data_root = cfg.get("root", "./data/uci_har")

        # Prefer the *direct* dataset zip; still works if user overrides url to the "bundle" zip.
        url = cfg.get(
            "url",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
        )

        os.makedirs(data_root, exist_ok=True)
        zip_path = os.path.join(data_root, os.path.basename(url) or "uci_har.zip")

        def find_uci_har_root(base_dir: str):
            """
            Return directory that contains:
            train/X_train.txt and test/X_test.txt
            Typically: <base_dir>/UCI HAR Dataset/
            """
            for root, dirs, files in os.walk(base_dir):
                if "train" in dirs and "test" in dirs:
                    if (
                        os.path.exists(os.path.join(root, "train", "X_train.txt"))
                        and os.path.exists(os.path.join(root, "test", "X_test.txt"))
                    ):
                        return root
            return None

        def extract_zip(zip_file: str, dst_dir: str):
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(dst_dir)

        # 1) If already extracted somewhere under data_root, use it.
        extracted_root = find_uci_har_root(data_root)

        # 2) If not found, download + extract.
        if extracted_root is None:
            if not os.path.exists(zip_path):
                print(f"[INFO] Downloading UCI HAR zip to: {zip_path}")
                urllib.request.urlretrieve(url, zip_path)

            print(f"[INFO] Extracting zip into: {data_root}")
            extract_zip(zip_path, data_root)

            # 2b) Handle the common nested-zip case (your current situation):
            #     outer zip extracts "UCI HAR Dataset.zip" which then contains "UCI HAR Dataset/train/..."
            inner_zip = os.path.join(data_root, "UCI HAR Dataset.zip")
            if os.path.exists(inner_zip):
                print(f"[INFO] Found nested zip, extracting: {inner_zip}")
                extract_zip(inner_zip, data_root)

            extracted_root = find_uci_har_root(data_root)

        if extracted_root is None:
            top = []
            try:
                top = os.listdir(data_root)
            except Exception:
                pass
            raise FileNotFoundError(
                "UCI HAR files not found after download/extract.\n"
                f"Looked for train/X_train.txt under: {data_root}\n"
                f"Top-level entries under root: {top}\n"
                "Expected to find a folder containing train/ and test/ with X_train.txt."
            )

        print(f"[INFO] Using UCI HAR root: {extracted_root}")

        def load_split(split: str):
            X_path = os.path.join(extracted_root, split, f"X_{split}.txt")
            y_path = os.path.join(extracted_root, split, f"y_{split}.txt")
            X = np.loadtxt(X_path).astype("float32")
            y = np.loadtxt(y_path).astype("int64") - 1  # 1..6 -> 0..5
            return X, y

        Xtr, ytr = load_split("train")
        Xte, yte = load_split("test")

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr).astype("float32")
        Xte = scaler.transform(Xte).astype("float32")

        train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        valid_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))

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


    def build_gtsrb_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        GTSRB (German Traffic Sign Recognition Benchmark)
        - images have varying sizes -> we resize to a fixed resolution
        - RGB images
        """
        cfg = self.params.get("dataset_cfg", {}).get("gtsrb", {})

        img_size = int(cfg.get("img_size", 32))  # 32 is a good default
        # Simple normalization (ImageNet-like is also fine); this is baseline-safe
        mean = cfg.get("mean", (0.5, 0.5, 0.5))
        std = cfg.get("std", (0.5, 0.5, 0.5))

        gtsrb_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_ds = datasets.GTSRB(
            root="./data",
            split="train",
            download=True,
            transform=gtsrb_transform,
        )
        test_ds = datasets.GTSRB(
            root="./data",
            split="test",
            download=True,
            transform=gtsrb_transform,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, valid_loader

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
        if dataset in ("uci_har", "har", "uci"):
            return self.build_uci_har_loaders()
        if dataset in ("gtsrb", "traffic_signs", "traffic-signs"):
            return self.build_gtsrb_loaders()

        raise ValueError(
            f"Unknown dataset: {dataset}. "
            "Supported: mnist, fashion_mnist, emnist_balanced, digits, blobs, gtsrb"
        )
