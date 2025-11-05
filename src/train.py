import os
import json
import torch
import uuid
import argparse
from typing import Dict
from utils import load_params
from dataloader import DataLoaderFactory
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
from models import create_autoencoder, BaseAE


@torch.no_grad()
def encode_latents(model, loader, device):
    """Return (Z, y) arrays for a dataset."""
    model.eval()
    Zs, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        outputs = model(xb)
        z = outputs["z"] if isinstance(outputs, dict) else outputs[1]
        Zs.append(z.cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(Zs, axis=0), np.concatenate(ys, axis=0)


def compute_knn_accuracy(model, train_loader, val_loader, device, k=5):
    """Train kNN on train latents and compute accuracy on both sets."""
    Ztr, ytr = encode_latents(model, train_loader, device)
    Zva, yva = encode_latents(model, val_loader, device)

    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf.fit(Ztr, ytr)

    train_acc = float(clf.score(Ztr, ytr))
    val_acc = float(clf.score(Zva, yva))
    return train_acc, val_acc


def train_one(model: "BaseAE", train_loader, val_loader, optimizer, device,
              epochs: int, loss_cfg: dict, k: int = 5, knn_every: int = 1):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    model.to(device)

    for ep in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss, _ = model.loss((xb, yb), outputs, loss_cfg)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # ---- Validation loss ----
        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                outputs = model(xb)
                loss, _ = model.loss((xb, yb), outputs, loss_cfg)
                vtotal += loss.item() * xb.size(0)
        val_loss = vtotal / len(val_loader.dataset)
        history["val_loss"].append(val_loss)

        # ---- Latent-space accuracy (kNN) ----
        if ep % knn_every == 0:
            train_acc, val_acc = compute_knn_accuracy(model,
                                                      train_loader,
                                                      val_loader, device, k)
        else:
            train_acc = val_acc = np.nan
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # ---- Log ----
        msg = (f"Epoch {ep:02d}/{epochs} | "
               f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
               f"train_acc={train_acc*100:.2f}%, val_acc={val_acc*100:.2f}%")
        print(msg)

    return history


def train_many(train_loader,
               val_loader,
               params: Dict,
               save: bool = True):
    device = torch.device(params["device"])

    # ---- Create run directory FIRST ----
    base_output_dir = Path(params["output_dir"])
    base_output_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex[:7]
    run_dir = base_output_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}

    for d in params["latent_dims"]:
        print(f"\n=== Training {params['ae_type']} with latent_dim={d} ===")
        model = create_autoencoder(
            params.get("ae_type","ae"),
            input_dim=int(np.prod(next(iter(train_loader))[0].shape[1:])),
            latent_dim=int(d),
            enc_layers=params["encoder_layers"],
            dec_layers=params["decoder_layers"],
            activation=params["activation"],
            noise_std=params.get("noise_std", 0.1),
        ).to(device)

        opt = torch.optim.Adam(model.parameters(),
                               lr=float(params["learning_rate"]),
                               weight_decay=float(params["weight_decay"]))

        history = train_one(
            model, train_loader, val_loader, opt, device,
            epochs=int(params["epochs"]),
            loss_cfg={
                "recon_loss": params.get("loss_fn","mse"),
                "beta": params.get("beta", 1.0),
                "l1_latent": params.get("l1_latent", 0.0),
                "contractive_lambda": params.get("contractive_lambda", 0.0),
            },
        )

        summaries[int(d)] = {"history": history}

        if save:
            # store each model inside the run directory:
            latent_dir = run_dir / f"ae_latent{d}"
            latent_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), latent_dir / "model.pt")
            (latent_dir / "history.json").write_text(json.dumps(
                {"history": history}, indent=2))

    # ---- save summary of all runs ----
    (run_dir / "summary_train.json").write_text(
        json.dumps(summaries, indent=2)
    )

    print(f"\nSaved everything to: {run_dir}")
    return summaries



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--dataset",
                        choices=["mnist", "digits"],
                        default="mnist")
 
    args = parser.parse_args()

    params = load_params(args.params)
    factory = DataLoaderFactory(params)

    train_loader, valid_loader = factory.build("mnist")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(valid_loader)}")

    train_many(train_loader, valid_loader, params)
