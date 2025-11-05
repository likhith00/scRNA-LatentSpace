
from typing import List
from torch import nn
from .base import BaseAE, get_activation, get_loss_fn, register_ae


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int,
                 latent_dim: int,
                 hidden_layers: List[int],
                 activation="relu"):
        super().__init__()
        layers = []
        prev = input_dim
        act = get_activation(activation)
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), act]
            prev = h
        layers += [nn.Linear(prev, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int,
                 hidden_layers: List[int], activation="relu"):
        super().__init__()
        layers = []
        prev = latent_dim
        act = get_activation(activation)
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), act]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        return self.net(z)

@register_ae("ae")
class VanillaAE(BaseAE):
    def __init__(self, input_dim: int, latent_dim: int,
                 enc_layers: List[int], dec_layers: List[int],
                 activation="relu", **kwargs):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, latent_dim, 
                                  enc_layers, activation)
        self.decoder = MLPDecoder(latent_dim, input_dim, 
                                  dec_layers, activation)

    def encode(self, x_flat):
        return self.encoder(x_flat), {}

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        z, _ = self.encode(x_flat)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "z": z}

    def loss(self, batch, outputs, loss_cfg):
        x, _ = batch
        x = x.view(x.size(0), -1)
        recon_loss_fn = get_loss_fn(loss_cfg.get("recon_loss", "mse"))
        recon = recon_loss_fn(outputs["x_hat"], x)
        return recon, {"recon": float(recon.detach().cpu().item())}
