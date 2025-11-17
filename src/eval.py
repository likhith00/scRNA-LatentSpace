import os, json, argparse
from pathlib import Path
from typing import Dict, Tuple

import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from models import create_autoencoder, get_loss_fn  # type: ignore
from utils import load_params
from dataloader import DataLoaderFactory  # type: ignore

try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False


def infer_input_dim(loader):
    xb, _ = next(iter(loader))
    return int(np.prod(xb.shape[1:]))

@torch.no_grad()
def evaluate_ae(model, loader, loss_fn, device):
    model.eval(); total = 0.0
    for xb, _ in loader:
        xb = xb.to(device).view(xb.size(0), -1)
        out = model(xb)
        x_hat = out["x_hat"]
        loss = loss_fn(x_hat, xb)
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def encode_dataset(model, loader, device):
    model.eval(); Zs=[]; ys=[]
    for xb, yb in loader:
        xb = xb.to(device).view(xb.size(0), -1)
        out = model(xb)
        Zs.append(out["z"].cpu().numpy()); ys.append(yb.numpy())
    return np.concatenate(Zs, axis=0), np.concatenate(ys, axis=0)


def load_models_for_run(params: Dict, input_dim: int, run_dir: Path):

    device = torch.device(params["device"])
    models = {}
    for d in params["latent_dims"]:
        ckpt = run_dir / f"ae_latent{d}" / "model.pt"
        if not ckpt.exists():
            print(f"WARNING: missing {ckpt}"); continue
        m = create_autoencoder(params.get("ae_type","ae"),
                               input_dim=input_dim, latent_dim=int(d),
                               enc_layers=params["encoder_layers"],
                               dec_layers=params["decoder_layers"],
                               activation=params["activation"]).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        models[int(d)] = m
    return models


def evaluate_all(models, loader, params: Dict, run_dir: Path):
    """
    Save *all* evaluation artifacts under the same run directory:
      run_dir/
        ae_latent{d}/latent_histograms_d{d}.png
        ae_latent{d}/latent_umap_scatter.png
        loss_vs_latent_dim.png
        summary_eval.json
    """
    device = torch.device(params["device"])
    loss_fn = get_loss_fn(params["loss_fn"])
    run_dir.mkdir(parents=True, exist_ok=True)
    test_losses = {}
    for d, m in models.items():
        print(f"Evaluating latent_dim={d}")
        loss = evaluate_ae(m, loader, loss_fn, device)
        test_losses[d] = float(loss)

        Z, y = encode_dataset(m, loader, device)

        Z = Z.astype(np.float32, copy=False)
        y = y.astype(np.int64, copy=False)

        latent_dir = run_dir / f"ae_latent{d}"
        latent_dir.mkdir(parents=True, exist_ok=True)
        np.save(latent_dir / "Z.npy", Z)
        np.save(latent_dir / "y.npy", y)

    (run_dir / "summary_eval.json").write_text(
        json.dumps({"test_losses": test_losses,
                    "loss_type": params["loss_fn"]},
                    indent = 2))
    print(f"\nSaved everything to: {run_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--dataset", choices=["mnist","digits"],
                        default="mnist")
    parser.add_argument("--run-dir")
    
    args = parser.parse_args()

    params = load_params(args.params)
    run_dir = Path(args.run_dir)
    dataset = args.dataset

    factory = DataLoaderFactory(params)

    train_loader, valid_loader = factory.build(dataset=dataset)
    # Load models
    input_dim = infer_input_dim(valid_loader)
    models = load_models_for_run(params, input_dim, run_dir)
    evaluate_all(models, valid_loader, params, run_dir)

    if not models:
        print("No models found under output_dir. Train first.")
        exit(0)
