import argparse, json, re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False


class AEVisualizer:
    """
    Visualization-only helper. Reads existing artifacts from a run directory:
      - Per-latent training curves from:   run_dir/ae_latent{d}/history.json
      - Loss-vs-dim from:                  run_dir/summary_eval.json
      - UMAPs from cached latents:         run_dir/ae_latent{d}/latents_d{d}.npz (Z,y)
                                           or Z.npy / y.npy (side-by-side)
    Saves figures back into the same locations.
    """

    def __init__(self, run_dir: Path, enable_umap: bool = True):
        self.run_dir = Path(run_dir)
        self.enable_umap = enable_umap

    def _latent_dir(self, d: int) -> Path:
        p = self.run_dir / f"ae_latent{d}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _read_history(self, d: int) -> Optional[Dict]:
        f = self._latent_dir(d) / "history.json"
        if not f.exists():
            return None
        try:
            data = json.loads(f.read_text())
            return data.get("history", data)
        except Exception:
            return None

    def _read_summary(self) -> Dict:
        f = self.run_dir / "summary_eval.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                pass
        return {}

    def _load_cached_latents(self, d: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Try multiple conventions without recomputing anything."""
        ld = self._latent_dir(d)
        # Preferred: combined npz
        npz = ld / f"latents_d{d}.npz"
        if npz.exists():
            with np.load(npz) as data:
                if "Z" in data and "y" in data:
                    return data["Z"], data["y"]
        # Fallback: separate files
        zn, yn = ld / "Z.npy", ld / "y.npy"
        if zn.exists() and yn.exists():
            return np.load(zn), np.load(yn)
        return None

    def plot_training_curves(self, d: int) -> None:
        hist = self._read_history(d)
        if not hist:
            print(f"[info] no history.json for d={d}, skipping curves.")
            return

        tl = hist.get("train_loss", [])
        vl = hist.get("val_loss", [])
        ta = hist.get("train_acc", [])
        va = hist.get("val_acc", [])

        # Loss curves
        if tl or vl:
            plt.figure(figsize=(6,4))
            if tl: plt.plot(tl, label="train_loss")
            if vl: plt.plot(vl, label="val_loss")
            plt.xlabel("Epoch"); plt.ylabel("Loss")
            plt.title(f"Loss curves (d={d})")
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(self._latent_dir(d) / "loss_curves.png", dpi=150)
            plt.close()

        # Accuracy curves
        def _has_real(vals):
            return bool(vals) and np.isfinite(np.array(vals, dtype=float)).any()
        if _has_real(ta) or _has_real(va):
            plt.figure(figsize=(6,4))
            if ta: plt.plot(ta, label="train_acc")
            if va: plt.plot(va, label="val_acc")
            plt.xlabel("Epoch"); plt.ylabel("Accuracy")
            plt.title(f"Accuracy curves (d={d})")
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(self._latent_dir(d) / "accuracy_curves.png", dpi=150)
            plt.close()

    def plot_loss_vs_dim(self) -> None:
        summary = self._read_summary()
        test_losses = summary.get("test_losses", {})
        if not test_losses:
            print("[info] no test_losses in summary_eval.json; skipping loss-vs-dim.")
            return
        # keys may be strings, normalize
        test_losses = {int(k): float(v) for k, v in test_losses.items()}
        dims = sorted(test_losses.keys())
        vals = [test_losses[d] for d in dims]
        plt.figure(figsize=(6,4))
        plt.plot(dims, vals, marker="o")
        plt.xlabel("Latent dimension"); plt.ylabel("Reconstruction loss")
        plt.title("Test loss vs latent dimension")
        plt.grid(True); plt.tight_layout()
        plt.savefig(self.run_dir / "loss_vs_latent_dim.png", dpi=150)
        plt.close()

    def plot_umap_if_available(self, d: int) -> None:
        if not self.enable_umap:
            return
        if not HAVE_UMAP:
            print("[info] UMAP not installed; skipping.")
            return
        latents = self._load_cached_latents(d)
        if latents is None:
            print(f"[info] no cached latents for d={d}; 
                  expected latents_d{d}.npz or Z.npy/y.npy. Skipping UMAP.")
            return
        Z, y = latents
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        Z2 = reducer.fit_transform(Z)
        plt.figure(figsize=(6,5))
        sc = plt.scatter(Z2[:,0], Z2[:,1], c=y, s=5, alpha=0.7)
        cbar = plt.colorbar(sc)
        cbar.set_label("Label")
        plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2"); plt.title(f"UMAP (d={d})");

        plt.tight_layout();
        plt.savefig(self._latent_dir(d) / "latent_umap_scatter.png", dpi=150);
        plt.close();

    def run(self, latent_dims: List[int]) -> None:
        for d in sorted(int(x) for x in latent_dims):
            self.plot_training_curves(d)
            self.plot_umap_if_available(d)
        self.plot_loss_vs_dim()
        print(f"[done] saved visualizations to: {self.run_dir}")


def _find_latest_run_dir(base: Path) -> Optional[Path]:
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and re.match(r"^run_[0-9a-f]{7}$", p.name)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", help="Path to run directory, e.g. outputs/run_ab12cde")
    ap.add_argument("--output-dir", default="outputs", help="Base outputs dir (used only if --run-dir not set)")
    ap.add_argument("--latent-dims", nargs="+", type=int, help="Latent dims to visualize (defaults to those present)")
    ap.add_argument("--umap", action="store_true", help="Plot UMAPs only if cached latents are available")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _find_latest_run_dir(Path(args.output_dir))
    if not run_dir or not run_dir.exists():
        print("[error] no valid run directory. Provide --run-dir or ensure an outputs/run_* exists.")
        raise SystemExit(1)

    # If dims not given, infer from existing ae_latent* folders
    if args.latent_dims:
        dims = args.latent_dims
    else:
        dims = []
        for p in run_dir.iterdir():
            if p.is_dir() and p.name.startswith("ae_latent"):
                try:
                    dims.append(int(p.name.replace("ae_latent","")))
                except ValueError:
                    pass
        if not dims:
            print("[error] no ae_latent* folders found in run dir.")
            raise SystemExit(1)

    viz = AEVisualizer(run_dir=run_dir, enable_umap=args.umap)
    viz.run(dims)
