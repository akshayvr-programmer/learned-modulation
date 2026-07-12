"""The (n_ch, k) channel autoencoder.

Transmitter = encoder. Constellation = latent space. Channel = noisy bottleneck.
Receiver = decoder. Trained end-to-end with cross-entropy over M = 2^k messages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """One-hot message -> power-normalized complex symbol(s).

    The input is one-hot, not the integer m. With one-hot input, Linear(M, h) selects
    a column of the weight matrix, so the first layer is a learnable lookup table:
    M free vectors, one per message. Feeding the integer instead would compute w*m + b,
    forcing message 2 to produce twice the pre-activation of message 1 -- a false
    ordering on what are arbitrary categorical labels.
    """

    def __init__(self, M: int, n_ch: int = 1, hidden: int = 64):
        super().__init__()
        self.M, self.n_ch = M, n_ch
        self.net = nn.Sequential(
            nn.Linear(M, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * n_ch),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = self.net(s)
        # Average power constraint: E[||x||^2] = n_ch, i.e. Es = 1 per complex use.
        #
        # This is the physics, not a numerical nicety. Without it the encoder drives
        # ||x|| -> infinity, drowns the noise, and training loss collapses to exactly
        # 0.0000 -- it "solves" communication by transmitting with infinite power.
        # The constellation that emerges is the equilibrium between two opposing
        # forces: spread points apart to reduce confusion, but stay inside the budget.
        power = x.pow(2).sum(dim=1).mean()
        return x * torch.sqrt(torch.tensor(float(self.n_ch), device=x.device) / power)


class Decoder(nn.Module):
    """Received symbol(s) -> logits over M messages.

    No softmax here: nn.CrossEntropyLoss applies log-softmax internally. Applying it
    twice is a silent bug that just flattens the gradients.

    After softmax, the outputs estimate the posterior P(m | r). Classical detection
    theory says the optimal receiver is the MAP detector, argmax_m P(m | r). Minimizing
    cross-entropy drives the softmax toward the true posterior, so the trained decoder
    IS a learned MAP detector -- and its decision regions are the Voronoi cells of the
    learned constellation.
    """

    def __init__(self, M: int, n_ch: int = 1, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n_ch, hidden), nn.ReLU(),
            nn.Linear(hidden, M),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class ChannelAutoencoder(nn.Module):
    def __init__(self, k: int, n_ch: int = 1, hidden: int = 64):
        super().__init__()
        self.k, self.M, self.n_ch = k, 2 ** k, n_ch
        self.encoder = Encoder(self.M, n_ch, hidden)
        self.decoder = Decoder(self.M, n_ch, hidden)

    def forward(self, m: torch.Tensor, channel, sigma: float) -> torch.Tensor:
        x = self.encoder(F.one_hot(m, self.M).float())
        r = channel(x, sigma)
        return self.decoder(r)

    @torch.no_grad()
    def constellation(self) -> torch.Tensor:
        """The learned constellation: the encoder applied to every possible message."""
        device = next(self.parameters()).device
        return self.encoder(torch.eye(self.M, device=device)).cpu()
