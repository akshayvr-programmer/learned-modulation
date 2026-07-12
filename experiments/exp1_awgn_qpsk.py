"""Experiment 1 — M=4 on AWGN. Does the autoencoder rediscover QPSK?

The point is NOT to beat classical theory. On AWGN, classical theory is provably
optimal, so there is nothing to beat. The point is to show the learned system lands ON
the optimum, which validates the method before we take it to channels where the
optimum is unknown.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baselines import qpsk, qpsk_ser_theory                      # noqa: E402
from train import train, block_error_rate, constellation_stats   # noqa: E402

EBN0 = np.arange(0, 11)

print("Training (n_ch=1, k=2) autoencoder on AWGN @ 7 dB ...")
model = train(k=2, n_ch=1, channel="awgn", train_ebn0_db=7.0, steps=10_000)

C = model.constellation().numpy()
stats = constellation_stats(C)
print(f"\n  mean energy   = {stats['mean_energy']:.4f}   (constraint: 1.0)")
print(f"  min distance  = {stats['min_distance']:.4f}   (QPSK optimum: {np.sqrt(2):.4f})")

print("\nEb/N0   learned BLER   QPSK theory")
rows = []
for db in EBN0:
    ae = block_error_rate(model, "awgn", float(db))
    th = float(qpsk_ser_theory(float(db)))
    rows.append({"ebn0_db": int(db), "learned_bler": ae, "qpsk_theory_ser": th})
    print(f"{db:2d} dB   {ae:.6f}      {th:.6f}")

json.dump({"constellation": C.tolist(), **stats, "bler": rows},
          open(ROOT / "results" / "exp1_awgn_qpsk.json", "w"), indent=2)

# --- learned constellation vs QPSK ------------------------------------------
Q = qpsk()
th = np.linspace(0, 2 * np.pi, 200)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(np.cos(th), np.sin(th), color="0.85", lw=1, zorder=0)   # power budget
ax.scatter(Q[:, 0], Q[:, 1], s=170, marker="s", facecolors="none",
           edgecolors="gray", lw=1.4, label="classical QPSK")
ax.scatter(C[:, 0], C[:, 1], s=110, c="crimson", zorder=3, label="learned")
for i, p in enumerate(C):
    ax.annotate(f"{i:02b}", p, xytext=(8, 6), textcoords="offset points", fontsize=9)
ax.axhline(0, color="0.93"); ax.axvline(0, color="0.93")
ax.set_aspect("equal"); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
ax.set_title("Learned constellation, AWGN")
fig.tight_layout(); fig.savefig(ROOT / "figures" / "learned_qpsk.png", dpi=160)

# --- BLER vs theory ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.semilogy(EBN0, [r["qpsk_theory_ser"] for r in rows], "k-", lw=1.6,
            label="QPSK theory (optimal ML detector)")
ax.semilogy(EBN0, [r["learned_bler"] for r in rows], "o", ms=6, color="crimson",
            label="learned autoencoder")
ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("block error rate")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
ax.set_title("Learned system matches the theoretical optimum")
fig.tight_layout(); fig.savefig(ROOT / "figures" / "exp1_bler.png", dpi=160)

# --- decoder decision regions (Voronoi cells, learned not derived) -----------
g = np.linspace(-2, 2, 400)
G = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
with torch.no_grad():
    lab = model.decoder(torch.tensor(G, dtype=torch.float32)).argmax(1).numpy()
fig, ax = plt.subplots(figsize=(5, 5))
ax.contourf(g, g, lab.reshape(400, 400), levels=[-.5, .5, 1.5, 2.5, 3.5],
            colors=["#dbe9f6", "#fde7e0", "#e2f0e2", "#f3e6f5"])
ax.scatter(C[:, 0], C[:, 1], s=110, c="k", zorder=3)
ax.set_aspect("equal"); ax.set_xlabel("I"); ax.set_ylabel("Q")
ax.set_title("Decision regions learned by the decoder")
fig.tight_layout(); fig.savefig(ROOT / "figures" / "exp1_regions.png", dpi=160)

print("\nfigures + results written.")
