
from typing import Dict, Callable
from torch import nn


def get_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU(0.2)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def get_loss_fn(name: str):
    n = (name or "mse").lower()
    if n == "mse":
        return nn.MSELoss()
    if n == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unsupported loss_fn: {name}")


class BaseAE(nn.Module):
    def loss(self, batch, outputs, loss_cfg):
        raise NotImplementedError


AE_REGISTRY: Dict[str, Callable] = {}


def register_ae(name: str):
    def deco(cls):
        AE_REGISTRY[name.lower()] = cls
        return cls
    return deco


def create_autoencoder(ae_type: str, **kwargs) -> BaseAE:
    key = (ae_type or "ae").lower()
    if key not in AE_REGISTRY:
        raise ValueError(f"Autoencoder type '{ae_type}' not found in registry.")
    return AE_REGISTRY[key](**kwargs)
