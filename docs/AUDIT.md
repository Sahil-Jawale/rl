# Code Audit Report: `Sahil-Jawale/rl`

> **Specification Reference:** `CHESS_ENDGAME_RL_REPLICATION_SPEC.md` (Section 3: Phase 1 Audit)  
> **Date:** October 2026  
> **Auditor:** DeepMind Antigravity Autonomous Coding Agent  

---

## Executive Summary

An audit of the `chess_endgame_rl` package was conducted against the 8 defect classes identified in `CHESS_ENDGAME_RL_REPLICATION_SPEC.md`.

| # | Item | Status | Key Evidence | Impact |
|---|---|:---:|---|---|
| 1 | **Dueling mean over legal actions only** | **N/A** | `agents/dqn.py:14-16` | Currently a monolithic 2-layer MLP (not dueling). Rule applies to Phase 4. |
| 2 | **Double-DQN target argmax restricted to legal moves** | **FAIL** | `agents/dqn.py:206-218` | Single DQN target $\max Q_{\text{target}}$ is used without Double-DQN decoupling. |
| 3 | **Perspective flip** | **PASS** / **N/A** | `env/endgame_env.py:86-172` | Single agent always plays White; $s'$ is from White's perspective. (Phase 2 will generalize). |
| 4 | **`terminated` vs `truncated`** | **FAIL** | `env/endgame_env.py:165`, `agents/dqn.py:214` | Move limit sets `done=True`, zeroing out bootstrapping $V(s')$ on timeout. |
| 5 | **Terminal reward sign, scale & shaping** | **FAIL** | `env/endgame_env.py:318-337` | Black King distance calculation is dead code ($0.0$). Positive edge score enables stall-farming. |
| 6 | **Legal-action mask applied to exploration** | **PASS** | `agents/dqn.py:169`, `q_learning.py:75`, `ppo.py:135` | Epsilon exploration and PPO softmax are strictly masked to legal actions. |
| 7 | **Reachable-state sampling** | **FAIL** | `env/endgame_env.py:244-250`, `configs/scenarios.yaml:7` | `king_distance <= 2` rejects legal distance-2 opposition positions. |
| 8 | **Horizon** | **PASS** | `configs/scenarios.yaml:5`, `env/endgame_env.py:165` | 50 moves (100 plies) provides >3x margin over KRK max DTM (32 plies). |

**Summary:** 4 FAIL, 3 PASS / N/A, 1 N/A (Phase 4 requirement).

---

## Detailed Findings

### 1. Dueling Mean Over Legal Actions Only
* **Verdict:** `N/A` (Current Codebase) / `MANDATORY` (Phase 4 Target)
* **Evidence:** [`chess_endgame_rl/agents/dqn.py:14-16`](file:///c:/Users/abhis/OneDrive/Desktop/rl/chess_endgame_rl/agents/dqn.py#L14-L16)
  ```python
  # Architecture:
  #   Input (193) -> Dense(256, ReLU) -> Dense(128, ReLU) -> Dense(36, Linear)
  ```
* **Analysis:** The existing DQN is a standard single-stream network outputting 36 raw Q-values. It does not possess separate Value $V(s)$ and Advantage $A(s, a)$ heads.
* **Phase 4 Requirement:** In [`models/dueling_net.py`](file:///c:/Users/abhis/OneDrive/Desktop/rl/chess_endgame_rl/models/dueling_net.py), advantage normalization must strictly average over *legal* actions:
  $$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\text{LEGAL}(s)|} \sum_{a' \in \text{LEGAL}(s)} A(s, a')\right)$$
  Averaging over all 4096 outputs would inject noise from ~4000 illegal, never-activated outputs into every legal Q-value.

---

### 2. Double-DQN Target Argmax Restricted to Legal Moves
* **Verdict:** `FAIL`
* **Evidence:** [`chess_endgame_rl/agents/dqn.py:206-218`](file:///c:/Users/abhis/OneDrive/Desktop/rl/chess_endgame_rl/agents/dqn.py#L206-L218)
  ```python
  h1, h2, current_q = self._forward(states, use_target=False)
  _, _, next_q = self._forward(next_states, use_target=True)

  # Compute target Q-values
  target_q = current_q.copy()
  for i in range(len(states)):
      masked_next_q = next_q[i].copy()
      masked_next_q[legal_masks[i] == 0] = -1e9
      if dones[i]:
          target_q[i, actions[i]] = rewards[i]
      else:
          target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(masked_next_q)
  ```
* **Analysis:**
  1. The target calculation uses standard single DQN: `np.max(masked_next_q)` where `next_q` comes entirely from the target network.
  2. Double DQN requires action selection via the **online** network:
     $$a^* = \operatorname{argmax}_{a' \in \text{LEGAL}(s')} Q_{\text{online}}(s', a')$$
     $$y = r + \gamma (1 - \text{terminated}) Q_{\text{target}}(s', a^*)$$
  3. Single DQN induces severe overestimation bias, leading the agent to overestimate dead-end or looping states.

---

### 3. Perspective Flip
* **Verdict:** `PASS` (for single-agent vs scripted defender)
* **Evidence:** [`chess_endgame_rl/env/endgame_env.py:86-172`](file:///c:/Users/abhis/OneDrive/Desktop/rl/chess_endgame_rl/env/endgame_env.py#L86-L172), [`chess_endgame_rl/env/encoding.py:11-15`](file:///c:/Users/abhis/OneDrive/Desktop/rl/chess_endgame_rl/env/encoding.py#L11-L15)
* **Analysis:**
  * The environment models a single-agent MDP where White is always the learning agent.
  * In `step(action_id)`, White moves, the defender executes Black's reply, and the returned $s'$ is the board state at the start of White's next turn (`board.turn == chess.WHITE`).
  * Therefore, both $s$ and $s'$ are always from White's perspective. No inconsistent sign negations occur.
  * *Note:* Phase 2 will replace this with canonical 18-plane side-to-move perspective encoding to support full symmetry and self-play.

---

### 4. `terminated` vs `truncated`
* **Verdict:** `FAIL`
* **Evidence:**
  * [`chess_endgame_rl/env/endgame_env.py:165-170`](file:///c:/Users/abhis/OneDrive/Desktop/rl/chess_endgame_rl/env/endgame_env.py#L165-L170):
    ```python
    if self.move_count >= self.move_cap * 2:  # move_cap is in full moves
        self.done = True
        reward += -3.0
        info["reason"] = "move_limit"
        return self._observation(), reward, True, info
    ```
  * [`chess_endgame_rl/agents/dqn.py:214-217`](file:///c:/Users/abhis/OneDrive/Desktop/rl/chess_endgame_rl/agents/dqn.py#L214-L217):
    ```python
    if dones[i]:
        target_q[i, actions[i]] = rewards[i]
    else:
        target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(masked_next_q)
    ```
* **Analysis:**
  * The environment conflates environment termination (`terminated`: checkmate, stalemate, insufficient material) with artificial episode cut-off (`truncated`: move limit).
  * In RL theory (Gymnasium standard), target updates must bootstrap on truncation:
    $$y = r + \gamma (1 - \text{terminated}) \max_{a'} Q(s', a')$$
  * Because `dones[i]` is True on timeout, DQN sets $y = r$, completely discarding $V(s')$. The network falsely learns that any state reached near the 50-move mark has an immediate value collapse to 0.

---

### 5. Terminal Reward Sign, Scale & Shaping
* **Verdict:** `FAIL`
* **Evidence:**
  1. **Dead-Code Shaping:** [`chess_endgame_rl/env/endgame_env.py:318-327`](file:///c:/Users/abhis/OneDrive/Desktop/rl/chess_endgame_rl/env/endgame_env.py#L318-L327)
     ```python
     bk_before = board_before.king(chess.BLACK)
     bk_after = board_after.king(chess.BLACK)
     ...
     cd_before = center_distance(bk_before)
     cd_after = center_distance(bk_after)
     reward += (cd_after - cd_before) * 0.15
     ```
     `_compute_reward` is executed immediately after White's move. The Black King has not moved, so `bk_before == bk_after` always. Hence `(cd_after - cd_before) * 0.15` is **identically 0.0**. The primary reward signal intended to teach White to drive the Black King to the rim never fired.
  2. **Perverse Stalling Reward Loop:** [`chess_endgame_rl/env/endgame_env.py:329-337`](file:///c:/Users/abhis/OneDrive/Desktop/rl/chess_endgame_rl/env/endgame_env.py#L329-L337)
     ```python
     def edge_score(sq: int) -> float:
         ...
         if edge_dist == 0:
             return 0.1
         return 0.0
     reward += edge_score(bk_after)
     ```
     When Black is on an edge square, White receives $+0.10$ per step with a step cost of $-0.02$, netting $+0.08$ per move. White is heavily incentivized to shuffle back and forth, farming $+0.08 \times 100 = +8.0$ reward, rather than delivering checkmate.
  3. **Configuration Discrepancy:** `chess_endgame_rl/env/rewards.py:13-17` defaults to $[+1.0, -0.5, 0.0]$, while `configs/scenarios.yaml:9-15` specifies $[+10.0, -5.0, -8.0]$.

---

### 6. Legal-Action Mask Applied to Exploration
* **Verdict:** `PASS`
* **Evidence:**
  * [`chess_endgame_rl/agents/dqn.py:169-170`](file:///c:/Users/abhis/OneDrive/Desktop/rl/chess_endgame_rl/agents/dqn.py#L169-L170):
    ```python
    if np.random.rand() < self.epsilon:
        return int(np.random.choice(legal_actions))
    ```
  * [`chess_endgame_rl/agents/q_learning.py:75-76`](file:///c:/Users/abhis/OneDrive/Desktop/rl/chess_endgame_rl/agents/q_learning.py#L75-L76):
    ```python
    if np.random.rand() < self.epsilon:
        return int(np.random.choice(legal_actions))
    ```
  * [`chess_endgame_rl/agents/ppo.py:135-136`](file:///c:/Users/abhis/OneDrive/Desktop/rl/chess_endgame_rl/agents/ppo.py#L135-L136):
    ```python
    masked_logits = self._mask_logits(logits, legal_mask)
    probs = self._softmax(masked_logits)
    ```
* **Analysis:** All agents strictly constrain exploratory action choices to the legal action subset. Illegal actions are never sampled during exploration.

---

### 7. Reachable-State Sampling
* **Verdict:** `FAIL`
* **Evidence:**
  * [`configs/scenarios.yaml:7`](file:///c:/Users/abhis/OneDrive/Desktop/rl/configs/scenarios.yaml#L7): `king_separation: 2`
  * [`chess_endgame_rl/env/endgame_env.py:244-250`](file:///c:/Users/abhis/OneDrive/Desktop/rl/chess_endgame_rl/env/endgame_env.py#L244-L250):
    ```python
    king_distance = chess.square_distance(
        board.king(chess.WHITE),
        board.king(chess.BLACK),
    )
    min_distance = self.scenario.get("king_separation", 2)
    if min_distance is not None and king_distance <= int(min_distance):
        continue
    ```
* **Analysis:**
  * `chess.square_distance` computes Chebyshev distance $\max(|\Delta r|, |\Delta f|)$.
  * In chess, kings cannot touch ($\text{distance} \ge 2$). Distance 2 (e.g. White King on $a1$, Black King on $a3$ or $c1$) is **entirely legal**, common, and represents direct opposition.
  * Discarding `king_distance <= 2` rejects all positions where kings are separated by a single square. This systematically excludes direct opposition and many 1-move-to-mate configurations from the training distribution.

---

### 8. Horizon
* **Verdict:** `PASS`
* **Evidence:**
  * [`configs/scenarios.yaml:5`](file:///c:/Users/abhis/OneDrive/Desktop/rl/configs/scenarios.yaml#L5): `move_cap: 50`
  * [`chess_endgame_rl/env/endgame_env.py:165`](file:///c:/Users/abhis/OneDrive/Desktop/rl/chess_endgame_rl/env/endgame_env.py#L165): `self.move_count >= self.move_cap * 2` (100 plies)
* **Analysis:**
  * Maximum Distance-to-Mate (DTM) in KRK is 32 plies (16 full moves); in KQK it is 20 plies (10 full moves).
  * A 50-move limit (100 plies) provides a 3.1x margin over KRK and 5.0x over KQK, allowing ample exploration latitude without truncating optimal or near-optimal lines.

---

## Action Plan for Phase 1 Fixes

1. **Add `test_audit_regressions.py`** defining tests for each failed item:
   * Test Double-DQN target calculation logic vs standard DQN.
   * Test `terminated` vs `truncated` separation in `EndgameEnv`.
   * Test reward shaping (no stall-farming, non-zero center-distance delta).
   * Test reachability of distance-2 king positions.
2. **Apply targeted fixes** in `chess_endgame_rl/env/endgame_env.py`, `chess_endgame_rl/agents/dqn.py`, and `configs/scenarios.yaml`.
3. Verify all tests pass with `pytest -q`.
