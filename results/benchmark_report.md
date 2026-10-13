# Benchmark report

## Summary

The current benchmark compares three model families under the same endgame setup and identical random seeds.

| Scenario | Algorithm | Mean reward | Episodes |
| --- | --- | ---: | ---: |
| krk | q-learning | -0.361 | 50 |
| krk | dqn | -0.421 | 20 |
| krk | ppo | -0.337 | 20 |
| knk | q-learning | 0.000 | 50 |
| knk | dqn | 0.000 | 20 |
| knk | ppo | 0.000 | 20 |

## Interpretation

- KR-K remains the primary learnable benchmark. The models are expected to learn a forced mate strategy, while the reward remains sparse and highly dependent on defender behavior.
- KN-K is a draw-control scenario. The benchmark is expected to remain near zero win rate, so the relevant metrics are survival and sanity checks rather than successful mate forcing.
- All comparisons use the same budget and the same random defender setup so that the learning curves are comparable across algorithms.

## Next steps

- Add stronger defender variants and tablebase-based KR-K evaluation.
- Extend the interface to compare three boards side by side and overlay move explanations.
- Expand experimentation with improved hyperparameters and equalized training steps.

