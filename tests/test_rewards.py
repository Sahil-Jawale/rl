from __future__ import annotations

import random

from chess_endgame_rl.env.rewards import PotentialShaping


def test_potential_shaping_telescoping_identity():
    """Verify sum_{t=0}^{T-1} gamma^t * F(s_t, s_{t+1}) == -Phi(s_0) to 1e-9."""
    gamma = 0.95
    c = 1.0
    dtm_max = 32
    shaper = PotentialShaping(gamma=gamma, c=c, dtm_max=dtm_max, enabled=True)

    rng = random.Random(42)
    # Simulate 50 trajectories of varying lengths
    for _ in range(50):
        length = rng.randint(5, 25)
        # Generate sequence of DTMs (e.g. 15, 14, 13, ..., 1, 0)
        dtm_sequence = [rng.randint(1, dtm_max) for _ in range(length)]
        # Final state is terminal checkmate (DTM 0)
        dtm_sequence.append(0)

        total_discounted_shaping = 0.0
        for t in range(len(dtm_sequence) - 1):
            s_t = dtm_sequence[t]
            s_next = dtm_sequence[t + 1]
            is_terminal = (t + 1 == len(dtm_sequence) - 1)
            f_t = shaper.shaping_reward(
                s_t,
                s_next,
                is_terminal=is_terminal,
                wdl_s=1,
                wdl_next=1,
            )
            total_discounted_shaping += (gamma ** t) * f_t

        phi_s0 = shaper.phi(dtm_sequence[0], is_terminal=False, wdl=1)
        expected = -phi_s0

        assert abs(total_discounted_shaping - expected) < 1e-9, (
            f"Telescoping identity violated: got {total_discounted_shaping}, expected {expected}"
        )


def test_potential_shaping_disabled_returns_zero():
    shaper = PotentialShaping(enabled=False)
    assert shaper.phi(10) == 0.0
    assert shaper.shaping_reward(10, 9) == 0.0


def test_potential_is_zero_for_draws_and_attacker_losses():
    shaper = PotentialShaping()
    assert shaper.phi(10, wdl=0) == 0.0
    assert shaper.phi(10, wdl=-1) == 0.0
    assert shaper.shaping_reward(10, 9, wdl_s=-1, wdl_next=-1) == 0.0
    assert shaper.shaping_reward(10, -1, wdl_s=1, wdl_next=0) == -shaper.phi(10)
