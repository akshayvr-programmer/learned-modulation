"""Channel models.

Every channel takes x of shape (B, 2*n_ch) laid out as [I, Q, I, Q, ...] and returns
a corrupted r of the same shape. All operations are differentiable in x: the noise is
sampled independently of x, so autograd treats it as an additive constant and
dr/dx = 1. Gradients therefore flow from the receiver's loss, through the channel,
into the transmitter's weights.

The channel is the ONLY thing that varies between experiments. Encoder, decoder, and
loss are held fixed.
"""

import numpy as np
import torch


def ebn0_to_sigma(ebn0_db: float, k: int) -> float:
    """Noise std-dev per real dimension, given Es = 1 per complex channel use.

        Es = 1  ->  Eb = Es / k  ->  N0 = Eb / (Eb/N0)  ->  sigma^2 = N0 / 2

    The final /2 is the one everyone forgets: a complex Gaussian with total variance
    N0 splits it evenly between the real and imaginary parts. Omitting it shifts every
    BER curve by 3 dB.
    """
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return float(np.sqrt(1.0 / (2.0 * k * ebn0)))


def awgn(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Additive white Gaussian noise. Classical theory is provably optimal here."""
    return x + sigma * torch.randn_like(x)


def clipping_amplifier(x: torch.Tensor, sigma: float, clip: float = 1.0) -> torch.Tensor:
    """Saturating power amplifier, followed by AWGN.

    Any symbol whose magnitude exceeds `clip` is compressed back to `clip`; phase is
    preserved. This is a crude but standard model of a PA driven near saturation.

    Note what this does to the optimization problem: the power constraint is on the
    AVERAGE energy, so the encoder is free to place some points far out. Here, doing
    so is punished -- but only above the threshold, and only in magnitude.

    `clip` is relative to unit average symbol energy, so clip = 1.0 means the
    amplifier saturates right at the average power level (aggressive). Larger values
    are gentler; clip >= ~1.5 is close to linear for most constellations.
    """
    xc = x.view(x.shape[0], -1, 2)                        # (B, n_ch, 2)
    mag = xc.norm(dim=-1, keepdim=True).clamp_min(1e-9)   # |x| per complex symbol
    scale = torch.clamp(mag, max=clip) / mag              # <= 1; only shrinks
    return awgn((xc * scale).reshape(x.shape), sigma)


def phase_noise(x: torch.Tensor, sigma: float, phase_std: float = 0.15) -> torch.Tensor:
    """Random per-symbol phase rotation (oscillator jitter), followed by AWGN."""
    xc = x.view(x.shape[0], -1, 2)
    theta = phase_std * torch.randn(xc.shape[0], xc.shape[1], device=x.device)
    c, s = torch.cos(theta), torch.sin(theta)
    I, Q = xc[..., 0], xc[..., 1]
    rot = torch.stack([I * c - Q * s, I * s + Q * c], dim=-1)
    return awgn(rot.reshape(x.shape), sigma)


CHANNELS = {
    "awgn": awgn,
    "clip": clipping_amplifier,
    "phase": phase_noise,
}
