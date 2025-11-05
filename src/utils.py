from pathlib import Path
from typing import Dict
import yaml
import torch


def load_params(path: str = "params.yaml") -> Dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"Parameter file '{path}' not found.")
    else:
        params = yaml.safe_load(open(path, "r"))
    # device handling
    if params.get("device", "auto") == "auto":
        params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    elif params["device"] == "cuda" and not torch.cuda.is_available():
        params["device"] = "cpu"

    return params
