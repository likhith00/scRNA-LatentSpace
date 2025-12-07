import os, json, argparse
from pathlib import Path
from typing import Dict, Tuple

import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
# --- NEU: Cluster- und Metrik-Imports ---
from sklearn.cluster import KMeans


from models import create_autoencoder, get_loss_fn  # type: ignore
from utils import load_params
from dataloader import DataLoaderFactory  # type: ignore
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)



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
    model.eval()
    Xs, Zs, ys = [], [], []
    for xb, yb in loader:
        xb = xb.to(device).view(xb.size(0), -1)  # flatten
        out = model(xb)
        Zs.append(out["z"].cpu().numpy())
        ys.append(yb.numpy())
        Xs.append(xb.cpu().numpy())  # store original flattened inputs
    X = np.concatenate(Xs, axis=0)
    Z = np.concatenate(Zs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, Z, y


def continuity(X: np.ndarray, Z: np.ndarray, n_neighbors: int = 15) -> float:
    """
    Continuity(k): do neighbors in original space remain neighbors in embedding?
    X: original high-dimensional data, shape (n_samples, n_features)
    Z: embedded/latent data, shape (n_samples, latent_dim)
    """
    n_samples = X.shape[0]
    if Z.shape[0] != n_samples:
        raise ValueError("X and Z must have same number of samples")

    if n_neighbors >= n_samples:
        raise ValueError("n_neighbors must be < n_samples")

    # k-NN in original space
    nn_X = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn_X.fit(X)
    neigh_ind_X = nn_X.kneighbors(return_distance=False)[:, 1:]  # drop self

    # k-NN in latent space
    nn_Z = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn_Z.fit(Z)
    neigh_ind_Z = nn_Z.kneighbors(return_distance=False)[:, 1:]  # drop self

    # Build neighbor sets
    neigh_set_X = [set(row) for row in neigh_ind_X]
    neigh_set_Z = [set(row) for row in neigh_ind_Z]

    # Ranks in latent space (for all points)
    # NOTE: O(n^2); may need subsampling for very large datasets
    D_Z = pairwise_distances(Z, metric="euclidean")
    order_Z = np.argsort(D_Z, axis=1)
    # rank_Z[i, j] = rank of j w.r.t i in latent space (0 = closest)
    rank_Z = np.argsort(order_Z, axis=1)

    # Continuity formula
    norm = 2.0 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))
    c_sum = 0.0

    for i in range(n_samples):
        # Points that are neighbors in original space but NOT in latent
        V_i = neigh_set_X[i] - neigh_set_Z[i]
        for j in V_i:
            r_ij = rank_Z[i, j]  # 0-based rank; j is never self, so >=1
            c_sum += (r_ij - n_neighbors)

    C = 1.0 - norm * c_sum
    return float(C)


def neighborhood_metrics(X: np.ndarray, Z: np.ndarray, n_neighbors: int = 15):
    """
    Returns (trustworthiness, continuity) for given X (original) and Z (latent).
    """
    t = float(trustworthiness(X, Z, n_neighbors=n_neighbors, metric="euclidean"))
    c = float(continuity(X, Z, n_neighbors=n_neighbors))
    return t, c


def load_models_for_run(params: Dict, input_dim: int, run_dir: Path):

    device = torch.device(params["device"])
    models = {}
    for d in params["latent_dims"]:
        ckpt = run_dir / f"ae_latent{d}" / "model.pt"
        if not ckpt.exists():
            print(f"WARNING: missing {ckpt}")
            continue
        m = create_autoencoder(
            params.get("ae_type", "ae"),
            input_dim=input_dim,
            latent_dim=int(d),
            enc_layers=params["encoder_layers"],
            dec_layers=params["decoder_layers"],
            activation=params["activation"]
        ).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        models[int(d)] = m
    return models


def evaluate_all(models, loader, params: Dict, run_dir: Path):
    """
    Save *all* evaluation artifacts under the same run directory:
      run_dir/
        ae_latent{d}/Z.npy
        ae_latent{d}/y.npy
        loss_vs_latent_dim.png   (optional, noch nicht implementiert)
        summary_eval.json        (inkl. ARI/NMI pro Latent-Dim)
    """
    device = torch.device(params["device"])
    loss_fn = get_loss_fn(params["loss_fn"])
    run_dir.mkdir(parents=True, exist_ok=True)

    test_losses: Dict[int, float] = {}
    cluster_metrics: Dict[int, Dict[str, float]] = {}  # --- NEU ---

    for d, m in models.items():
        print(f"Evaluating latent_dim={d}")
        loss = evaluate_ae(m, loader, loss_fn, device)
        test_losses[d] = float(loss)

        X, Z, y = encode_dataset(m, loader, device)

        X = X.astype(np.float32, copy=False)
        Z = Z.astype(np.float32, copy=False)
        y = y.astype(np.int64, copy=False)
        
        latent_dir = run_dir / f"ae_latent{d}"
        latent_dir.mkdir(parents=True, exist_ok=True)
        np.save(latent_dir / "X.npy", X)  
        np.save(latent_dir / "Z.npy", Z)
        np.save(latent_dir / "y.npy", y)

        n_clusters = len(np.unique(y))
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=0
        )
        y_pred = kmeans.fit_predict(Z)

        ari = adjusted_rand_score(y, y_pred)
        nmi = normalized_mutual_info_score(y, y_pred)
        t, c = neighborhood_metrics(X, Z, n_neighbors=params.get("n_neighbors", 15))
        sil = silhouette_score(Z, y_pred, metric="euclidean")
        db = davies_bouldin_score(Z, y_pred)
        ch = calinski_harabasz_score(Z, y_pred)
        corr, red_metrics = compute_latent_correlation_and_redundancy(Z)
        np.save(latent_dir / "corr.npy", corr)

        cluster_metrics[d] = {
            "ARI": float(ari),
            "NMI": float(nmi),
            "trustworthiness": t,
            "continuity": c,
            "silhouette": float(sil),
            "davies_bouldin": float(db),
            "calinski_harabasz": float(ch),
            "avg_abs_corr": red_metrics["avg_abs_corr"],
            "effective_dim_95var": red_metrics["effective_dim_95var"],
        }

        print(f"  latent_dim={d}: loss={loss:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}, Trustworthiness={t:.4f}, Continuity={c:.4f}, Silhouette={sil:.4f}, DB={db:.4f}, CH={ch:.4f}")

    # summary_eval.json jetzt mit Cluster-Metriken
    summary = {
        "test_losses": test_losses,
        "cluster_metrics": cluster_metrics,
        "loss_type": params["loss_fn"],
    }

    (run_dir / "summary_eval.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nSaved everything to: {run_dir}")

def compute_latent_correlation_and_redundancy(Z):
    """
    Z: (n_samples, d)
    Returns:
      corr: (d, d) correlation matrix
      metrics: dict with simple redundancy measures
    """
    import numpy as np
    from sklearn.decomposition import PCA

    # d x d correlation between latent dimensions
    corr = np.corrcoef(Z, rowvar=False)

    # average absolute off-diagonal correlation
    off_diag = np.abs(corr[np.triu_indices_from(corr, k=1)])
    avg_abs_corr = float(off_diag.mean()) if off_diag.size > 0 else 0.0

    # effective dimensionality via PCA (95% explained variance)
    pca = PCA().fit(Z)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    eff_dim = int(np.searchsorted(cumulative, 0.95) + 1)

    metrics = {
        "avg_abs_corr": avg_abs_corr,
        "effective_dim_95var": eff_dim,
    }
    return corr, metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--dataset", choices=["mnist", "digits"],
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
