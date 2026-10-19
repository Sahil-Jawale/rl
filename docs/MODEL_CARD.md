# Model card: endgame Dueling network

## Model details

The model is a convolutional Dueling network consuming 18x8x8 side-to-move chess planes and producing Q-values for 4096 `from*64+to` actions. Its value head estimates the state value; its advantage head estimates action preferences. When a legal-action mask is supplied, the advantage mean and action selection are restricted to legal moves.

## Intended use

This experimental model is intended for KQvK and KRvK endgame research, supervised warm-start experiments, and optional RL fine-tuning. It is not a general chess engine and does not use search at inference time.

## Training and evaluation

Supervised labels and DTM metrics come from the repository's exact three-piece retrograde solver. RL fine-tuning is a separate phase using terminal rewards, potential shaping, and an anchor loss.

The KRvK warm-start checkpoint was trained for five of the configured ten supervised epochs (seed 7) and selected at epoch 5. On its 17,517-position validation split it achieved 94.19% optimal moves, 99.46% moves that preserve a forced win, and 0.54% moves that turn a win into a draw. The bounded Double-DQN run started from this checkpoint, used the optimal defender, and early-stopped after 1,000 environment steps (79 episodes) as validation move quality degraded. Its retained checkpoint was the best observed RL snapshot.

Protocol v1 evaluation used seed 7, 10,000 held-out positions, and 150 games per defender from the held-out KRvK winning positions:

| Model | Optimal moves | Win-to-draw | Wins vs random | Wins vs tablebase-optimal |
|---|---:|---:|---:|---:|
| Tablebase warm-start | 94.06% | 0.52% | 96.0% | 82.67% |
| Warm-start + Double DQN | 91.45% | 0.48% | 80.67% | 56.0% |

The fine-tuning run did not improve over the supervised warm start, so the UI defaults to the warm-start checkpoint and keeps the RL checkpoint selectable for comparison. These are single-seed results on one three-piece endgame, not an Elo rating or evidence of full-chess strength. The game sample is limited, and performance should be rechecked across additional seeds before making broader claims.

## Limitations and risks

- Coverage is limited to KQvK and KRvK; larger material is unsupported.
- The model's labels and evaluation depend on exact solver access.
- Only queen promotions are representable in the action scheme.
- Performance can differ from tablebase-optimal play; report draws and losses rather than presenting the model as a reliable chess player.
- A high held-out move accuracy does not replace end-to-end game evaluation against both random and optimal defenders.
