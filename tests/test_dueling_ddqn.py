from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chess_endgame_rl.agents.dueling_ddqn import DuelingDoubleDQN


def test_double_dqn_targets_select_online_best_legal_action():
    rewards = torch.tensor([1.0, 2.0])
    terminated = torch.tensor([False, True])
    online = torch.tensor([[5.0, 100.0, 9.0], [1.0, 2.0, 3.0]])
    target = torch.tensor([[30.0, 900.0, 70.0], [10.0, 20.0, 30.0]])
    masks = torch.tensor([[True, False, True], [True, True, True]])

    values = DuelingDoubleDQN.double_dqn_targets(
        rewards, terminated, online, target, masks, gamma=0.5
    )

    torch.testing.assert_close(values, torch.tensor([36.0, 2.0]))


def test_double_dqn_rejects_nonterminal_empty_legal_mask():
    with pytest.raises(ValueError, match="at least one legal action"):
        DuelingDoubleDQN.double_dqn_targets(
            torch.zeros(1),
            torch.tensor([False]),
            torch.zeros((1, 2)),
            torch.zeros((1, 2)),
            torch.zeros((1, 2), dtype=torch.bool),
            gamma=0.9,
        )


def test_ddqn_replay_rejects_illegal_current_action():
    agent = DuelingDoubleDQN(
        batch_size=1, replay_capacity=1, hidden_dim=32, seed=0
    )
    state = np.zeros((18, 8, 8), dtype=np.float32)
    mask = np.zeros(4096, dtype=np.bool_)
    mask[17] = True
    with pytest.raises(ValueError, match="must be .* legal"):
        agent.remember(state, 18, 0.0, mask, state, False, mask)


def test_agent_explores_only_legal_moves_and_trains_with_anchor():
    agent = DuelingDoubleDQN(
        batch_size=1,
        replay_capacity=2,
        target_update_frequency=1,
        anchor_weight=0.1,
        seed=5,
        hidden_dim=32,
    )
    state = np.zeros((18, 8, 8), dtype=np.float32)
    mask = np.zeros(4096, dtype=np.bool_)
    mask[[17, 211]] = True

    for _ in range(10):
        assert agent.select_action(state, mask) in (17, 211)

    action = agent.select_action(state, mask, explore=False)
    agent.remember(state, action, 0.0, mask, state, False, mask)
    anchor_batch = {
        "state": torch.from_numpy(state).unsqueeze(0),
        "legal_mask": torch.from_numpy(mask).unsqueeze(0),
        "target_distribution": torch.nn.functional.one_hot(
            torch.tensor([action]), num_classes=4096
        ).to(torch.float32),
    }
    metrics = agent.train_step(anchor_batch)
    assert metrics is not None
    assert np.isfinite(list(metrics.values())).all()
    assert agent.steps == 1
