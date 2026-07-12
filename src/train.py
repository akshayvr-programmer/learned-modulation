"""Training and evaluation."""

import numpy as np
import torch
import torch.nn as nn

from channels import CHANNELS, ebn0_to_sigma
from models import ChannelAutoencoder


def train(k: int = 2, n_ch: int = 1, channel: str = "awgn",
          train_ebn0_db: float = 7.0, steps: int = 10_000, batch: int = 1024,
          lr: float = 1e-3, seed: int = 0, verbose: bool = True,
          **channel_kwargs) -> ChannelAutoencoder:
    """Train a channel autoencoder end-to-end.

    One optimizer holds BOTH networks' parameters -- transmitter and receiver are
    co-designed, not standardized separately as in a classical system.

    Data is generated on the fly: fresh messages and fresh noise every batch. There is
    no dataset, no train/test split, and no overfitting -- an unusual luxury.

    Loss will NOT converge to zero. It settles into a noisy band set by the channel's
    irreducible (Bayes) error at the training SNR. A loss of exactly 0.0000 means the
    power constraint is broken and the encoder is cheating.
    """
    torch.manual_seed(seed)
    model = ChannelAutoencoder(k, n_ch)
    ch_fn = CHANNELS[channel]
    ch = lambda x, s: ch_fn(x, s, **channel_kwargs)
    sigma = ebn0_to_sigma(train_ebn0_db, k)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    lossf = nn.CrossEntropyLoss()

    for step in range(steps):
        m = torch.randint(0, model.M, (batch,))
        loss = lossf(model(m, ch, sigma), m)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step + 1) % (steps // 5) == 0:
            print(f"  step {step + 1:6d}   loss {loss.item():.4f}")

    return model


@torch.no_grad()
def block_error_rate(model: ChannelAutoencoder, channel: str, ebn0_db: float,
                     n_test: int = 500_000, seed: int = 123,
                     **channel_kwargs) -> float:
    """P(message decoded incorrectly). For a single-symbol code this is the SER."""
    torch.manual_seed(seed)
    ch_fn = CHANNELS[channel]
    ch = lambda x, s: ch_fn(x, s, **channel_kwargs)
    sigma = ebn0_to_sigma(ebn0_db, model.k)

    errors = 0
    for i in range(0, n_test, 100_000):
        b = min(100_000, n_test - i)
        m = torch.randint(0, model.M, (b,))
        errors += (model(m, ch, sigma).argmax(dim=1) != m).sum().item()
    return errors / n_test


def constellation_stats(C: np.ndarray) -> dict:
    """Geometry of a constellation: energy, radii, angles, minimum distance."""
    M = len(C)
    energies = (C ** 2).sum(axis=1)
    radii = np.linalg.norm(C, axis=1)
    angles = np.degrees(np.arctan2(C[:, 1], C[:, 0])) % 360
    dists = [float(np.linalg.norm(C[i] - C[j]))
             for i in range(M) for j in range(i + 1, M)]
    return {
        "mean_energy": float(energies.mean()),
        "min_distance": float(min(dists)),
        "radii_sorted": np.sort(radii).round(3).tolist(),
        "radii": radii.round(3).tolist(),
        "angles": angles.round(1).tolist(),
    }
