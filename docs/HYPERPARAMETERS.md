# Hyperparameter record

The values below are implementation defaults, not sweep-selected settings. No three-seed parameter sweep has been run, so they must not be described as empirically optimal.

| Stage / parameter | Current default | How selected / status |
|---|---:|---|
| Supervised objective | Cross-entropy over optimal-move set | Default; Q regression is also implemented as an ablation. |
| Supervised epochs | 10 | Script default only; not yet selected by a validation sweep. |
| Supervised batch size | 64 | Script default only. |
| Supervised learning rate | 3e-4 | Script default only. |
| Validation fraction | 10% | Deterministic position-level split, seed 0 by default. |
| RL gamma | 0.99 | Script default; each step is an attacker move plus defender reply. |
| RL learning rate | 1e-4 | Script default only. |
| RL epsilon | 0.05 | Low exploration default to protect the supervised initialization. |
| RL anchor weight | 0.1 | Script default; anchor loss can be disabled with `--anchor-weight 0`. |
| RL replay batch size | 64 | Agent default. |
| Target-network hard sync | 1,000 updates | Agent default. |
| Replay capacity | 50,000 transitions | Agent default. |
| Potential shaping | Enabled | Disable with `--no-shaping` for an ablation. |
| Early-stop patience | 5 consecutive increases in validation suboptimal rate | Script default. |

Runs save their configuration and measured validation history in a metrics JSON alongside the checkpoint. Required sweeps, multiple seeds, learning curves, and final selected values remain outstanding.
