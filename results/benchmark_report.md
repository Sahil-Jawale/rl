# Benchmark report

## Summary

Each model is trained with the same episode budget and evaluated greedily on reproducibly seeded games.

| Scenario | Algorithm | Evaluation games | Checkmates | Checkmate rate | Draw rate | Timeout rate | Mean reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| krk | q-learning | 200 | 2 | 1.0% | 5.5% | 35.0% | -3.662 |
| krk | dqn | 200 | 2 | 1.0% | 2.0% | 45.0% | -4.073 |
| krk | ppo | 200 | 0 | 0.0% | 3.0% | 72.0% | -3.610 |

## Benchmark configuration

See benchmark_summary.json for per-seed metrics, training outcomes, model paths, and defender settings.

