# Problem definition and RL formulation

## Problem statement

Train an agent to convert won three-piece chess endgames into checkmate, using K+R vs K (`krk`) as the primary scenario and K+Q vs K (`kqk`) as the secondary scenario. K+N vs K (`knk`) remains a zero-win sanity case. The learned policy is evaluated against scripted defenders and an exact retrograde-analysis solver.

## Relevance and justification

Endgame conversion is a fully observable sequential decision problem with sparse terminal feedback and a large nominal action space containing many illegal moves. Exact endgame tablebases make it possible to measure an agent's move quality against ground truth, a useful experimental advantage for studying credit assignment and constrained action selection.

A tablebase already solves these positions exactly. This project therefore does not claim that reinforcement learning is the best way to play these chess endgames. The tablebase is an oracle for labels, potential values, and evaluation; learned policies demonstrate how value/policy approximation behaves on a domain with known answers. A learned model could generalize where a complete tablebase is infeasible, but this small-material project does not demonstrate that scaling claim.

## MDP formulation

| Element | Definition |
|---|---|
| Environment | Three-piece chess endgame: a White attacker against a random or tablebase-optimal Black defender. Transitions include one attacker move and, if the game remains active, one defender reply. |
| State | 18 binary 8x8 planes in side-to-move canonical perspective: own/opponent pieces, side-to-move, castling rights, and en-passant square. |
| Action | `from_square * 64 + to_square`, giving 4096 action indices. Only legal moves are selectable; promotions are queen-only in scope. |
| Reward | RL fine-tuning uses +1 for attacker checkmate, -1 for attacker checkmated, 0 for draws, and a small non-terminal step cost. Optional shaping uses `gamma * Phi(s') - Phi(s)`, where `Phi(s) = 1 - min(DTM, DTM_max)/DTM_max` for nonterminal attacker-winning states and zero for draws, attacker-losing states, and terminal states. |
| Discount | `gamma=0.99` by default in supervised Q-targets and RL fine-tuning; each transition is one attacker/defender turn. |
| Episode end | Checkmate, stalemate, insufficient material, claimable draw, or a 100-ply training/evaluation horizon. True terminal states and horizon truncations are stored separately for bootstrapping. |

## Measurement

Report W/D/L and win rate separately. Measure attacker move quality against tablebase-optimal actions, and report the rate of moves that turn a solved win into a draw. Do not use `(W + 0.5*D)/N` as the headline win metric. No full training/evaluation results are claimed until the checkpoint has been trained and evaluated under the recorded protocol.
