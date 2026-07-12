"""Classical baselines.

The critical design decision in this file: every baseline is SIMULATED through the
same channel object under test, rather than read off a closed-form BER formula.

On AWGN the two agree, so the formula is included for validation. On a nonlinear
channel they do NOT agree -- the formula assumes a linear amplifier and would
silently flatter the classical system, making the learned system look better than it
is. Any claim of the form "the autoencoder beats QAM on channel X" is only defensible
if the baseline was actually run through channel X.
"""

from itertools import combinations

import numpy as np
import torch
from scipy.special import erfc

from channels import CHANNELS, ebn0_to_sigma


# --- constellations, all normalized to unit average symbol energy -------------

def _unit_energy(C: np.ndarray) -> np.ndarray:
    return C / np.sqrt((C ** 2).sum(axis=1).mean())


def qpsk() -> np.ndarray:
    return _unit_energy(np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float))


def qam16() -> np.ndarray:
    return _unit_energy(np.array([[i, j] for i in (-3, -1, 1, 3) for j in (-3, -1, 1, 3)],
                                 dtype=float))


def psk(M: int) -> np.ndarray:
    th = np.arange(M) * 2 * np.pi / M
    return np.stack([np.cos(th), np.sin(th)], axis=1)


def apsk(inner: int, outer: int, ratio: float) -> np.ndarray:
    a = np.arange(inner) * 2 * np.pi / inner
    b = np.arange(outer) * 2 * np.pi / outer
    C = np.vstack([np.stack([np.cos(a), np.sin(a)], 1),
                   ratio * np.stack([np.cos(b), np.sin(b)], 1)])
    return _unit_energy(C)


def apsk_best(inner: int, outer: int) -> tuple:
    """APSK with the ring-radius ratio chosen to maximize minimum distance."""
    best = (0.0, None, None)
    for ratio in np.linspace(1.2, 5.0, 500):
        C = apsk(inner, outer, ratio)
        d = min_distance(C)
        if d > best[0]:
            best = (d, ratio, C)
    return best[2], best[1], best[0]


# --- geometry -----------------------------------------------------------------

def min_distance(C: np.ndarray) -> float:
    return min(float(np.linalg.norm(C[i] - C[j]))
               for i, j in combinations(range(len(C)), 2))


# --- theory (AWGN only; used to validate the simulator) -----------------------

def qfunc(z):
    return 0.5 * erfc(np.asarray(z) / np.sqrt(2))


def qpsk_ber_theory(ebn0_db):
    """Exact BER for Gray-coded QPSK on AWGN. The curve the simulator must reproduce."""
    return qfunc(np.sqrt(2 * 10.0 ** (np.asarray(ebn0_db) / 10.0)))


def qpsk_ser_theory(ebn0_db):
    return 1 - (1 - qpsk_ber_theory(ebn0_db)) ** 2


# --- simulation through ANY channel -------------------------------------------

@torch.no_grad()
def symbol_error_rate(C: np.ndarray, channel: str, ebn0_db: float, k: int,
                      n_test: int = 500_000, seed: int = 123,
                      **channel_kwargs) -> float:
    """Simulate a fixed constellation through `channel` with nearest-neighbour detection.

    Nearest-neighbour detection is the ML detector for AWGN. On a nonlinear channel it
    is no longer optimal -- but it is what a classical receiver actually does, which is
    exactly the comparison we want.
    """
    torch.manual_seed(seed)
    ch_fn = CHANNELS[channel]
    pts = torch.tensor(C, dtype=torch.float32)
    sigma = ebn0_to_sigma(ebn0_db, k)

    errors = 0
    for i in range(0, n_test, 100_000):
        b = min(100_000, n_test - i)
        m = torch.randint(0, len(C), (b,))
        r = ch_fn(pts[m], sigma, **channel_kwargs)
        errors += (torch.cdist(r, pts).argmin(dim=1) != m).sum().item()
    return errors / n_test


def benchmark_16(verbose: bool = True) -> dict:
    """Minimum distance of standard 16-point constellations at unit average energy."""
    rows = {"16-QAM": min_distance(qam16()), "16-PSK": min_distance(psk(16))}
    for inner, outer in [(4, 12), (5, 11), (6, 10)]:
        _, ratio, d = apsk_best(inner, outer)
        rows[f"APSK {inner}+{outer}"] = d
    if verbose:
        for name, d in sorted(rows.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<14s} min distance = {d:.4f}")
    return rows
