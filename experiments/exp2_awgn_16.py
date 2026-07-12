"""Experiment 2 — M=16 on AWGN.

Two questions:
  1. What geometry does the network converge to?
  2. Is it any good? Benchmarked against 16-QAM, 16-PSK and radius-optimized APSK,
     on BOTH minimum distance and (the metric that actually matters) block error rate.

Minimum distance is only a proxy. By the union bound, P(err) ~ N_min * Q(d_min / 2*sigma):
the NUMBER of nearest neighbours matters too, so a constellation with slightly smaller
d_min but fewer confusable neighbours can still win on error rate. Both are reported.

The seed sweep is not optional. 16-point packing has many local optima; a single pretty
run tells you nothing about whether the result is reproducible.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baselines import (qam16, psk, apsk_best, min_distance,       # noqa: E402
                       symbol_error_rate, benchmark_16)
from train import train, block_error_rate, constellation_stats    # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
TRAIN_DB = 12.0
EVAL_DB = [8, 10, 12, 14]

print("Classical 16-point benchmark (unit average energy):")
bench = benchmark_16()

print(f"\nTraining M=16 autoencoder, {len(SEEDS)} seeds @ {TRAIN_DB} dB ...")
runs = []
for seed in SEEDS:
    model = train(k=4, n_ch=1, channel="awgn", train_ebn0_db=TRAIN_DB,
                  steps=20_000, seed=seed, verbose=False)
    C = model.constellation().numpy()
    st = constellation_stats(C)
    bler = {db: block_error_rate(model, "awgn", float(db)) for db in EVAL_DB}
    runs.append({"seed": seed, "constellation": C.tolist(), **st,
                 "bler": {str(k_): v for k_, v in bler.items()}})
    print(f"  seed {seed}:  min dist = {st['min_distance']:.4f}   "
          f"mean E = {st['mean_energy']:.3f}   BLER@12dB = {bler[12]:.5f}")

d_all = [r["min_distance"] for r in runs]
print(f"\nSeed sensitivity: min distance {np.mean(d_all):.4f} +/- {np.std(d_all):.4f} "
      f"(range {min(d_all):.4f} - {max(d_all):.4f})")
print(f"16-QAM reference: {bench['16-QAM']:.4f}")

# --- the comparison that actually decides it: BLER vs 16-QAM ----------------
best = max(runs, key=lambda r: r["min_distance"])
print(f"\nBLER, best seed ({best['seed']}) vs classical 16-QAM:")
print("Eb/N0   learned    16-QAM     verdict")
bler_rows = []
for db in EVAL_DB:
    ae = best["bler"][str(db)]
    qam = symbol_error_rate(qam16(), "awgn", float(db), k=4)
    verdict = "learned" if ae < qam else "16-QAM"
    bler_rows.append({"ebn0_db": db, "learned": ae, "qam16": qam})
    print(f"{db:2d} dB   {ae:.5f}    {qam:.5f}    {verdict} wins")

json.dump({"benchmark_min_distance": bench, "runs": runs, "bler_vs_qam16": bler_rows},
          open(ROOT / "results" / "exp2_awgn_16.json", "w"), indent=2)

# --- plot every seed side by side -------------------------------------------
fig, axes = plt.subplots(1, len(SEEDS), figsize=(3.1 * len(SEEDS), 3.4))
for ax, r in zip(np.atleast_1d(axes), runs):
    C = np.array(r["constellation"])
    rad = np.linalg.norm(C, axis=1)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="0.9", lw=1, zorder=0)
    ax.scatter(C[:, 0], C[:, 1], s=45, c=rad, cmap="coolwarm", zorder=3)
    ax.set_aspect("equal"); ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"seed {r['seed']}\n$d_{{min}}$={r['min_distance']:.3f}", fontsize=9)
fig.suptitle("M=16 learned constellations across seeds "
             f"(16-QAM reference: {bench['16-QAM']:.3f})", fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "exp2_seed_sweep.png", dpi=160)
print("\nfigures + results written.")
