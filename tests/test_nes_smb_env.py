from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rl_zoo3.nes_smb_env import (
    RewardConfig,
    build_metrics_from_ram,
    compute_reward_transition,
    derive_termination_reason,
)


def _make_ram() -> np.ndarray:
    return np.zeros(0x800, dtype=np.uint8)


def test_build_metrics_from_ram():
    ram = _make_ram()
    ram[0x006D] = 1
    ram[0x0086] = 32
    ram[0x0755] = 12
    ram[0x075F] = 0
    ram[0x075C] = 0
    ram[0x0770] = 1
    ram[0x07F8] = 3
    ram[0x07F9] = 9
    ram[0x07FA] = 7

    metrics = build_metrics_from_ram(ram)

    assert metrics.progress == 288
    assert metrics.level_progress == 268
    assert metrics.world_display == 1
    assert metrics.level_display == 1
    assert metrics.timer == 397
    assert metrics.campaign_progress == 268


def test_compute_reward_transition_progress_and_coin():
    ram_prev = _make_ram()
    ram_curr = _make_ram()

    ram_prev[0x0086] = 40
    ram_prev[0x0755] = 40
    ram_prev[0x075A] = 2
    ram_prev[0x075E] = 0
    ram_prev[0x0770] = 1
    ram_prev[0x000E] = 8
    ram_prev[0x07F8] = 4

    ram_curr[:] = ram_prev
    ram_curr[0x0086] = 52
    ram_curr[0x0755] = 50
    ram_curr[0x075E] = 1
    ram_curr[0x07F8] = 3

    previous = build_metrics_from_ram(ram_prev)
    current = build_metrics_from_ram(ram_curr)
    transition = compute_reward_transition(previous, current, RewardConfig(stagnation_steps=240), stagnation_steps=5)

    assert transition.reward > 0
    assert transition.stagnation_steps == 0
    assert "progress" in transition.reward_parts
    assert transition.reward_parts["coins"] == 1.0
    assert transition.reward_parts["time_penalty"] == -0.01


def test_compute_reward_transition_death_and_stagnation():
    ram_prev = _make_ram()
    ram_curr = _make_ram()

    ram_prev[0x0086] = 80
    ram_prev[0x0755] = 80
    ram_prev[0x075A] = 2
    ram_prev[0x0770] = 1
    ram_prev[0x000E] = 8

    ram_curr[:] = ram_prev
    ram_curr[0x075A] = 1
    ram_curr[0x000E] = 6

    previous = build_metrics_from_ram(ram_prev)
    current = build_metrics_from_ram(ram_curr)
    transition = compute_reward_transition(previous, current, RewardConfig(stagnation_steps=2), stagnation_steps=2)

    assert transition.reward < 0
    assert transition.reward_parts["death_penalty"] == -25.0
    assert "stagnation_penalty" in transition.reward_parts


def test_derive_termination_reason():
    ram = _make_ram()
    ram[0x075A] = 1
    ram[0x000E] = 6
    ram[0x0770] = 1
    metrics = build_metrics_from_ram(ram)

    terminated, truncated, reason = derive_termination_reason(
        metrics,
        RewardConfig(stagnation_steps=240),
        level_complete=False,
        terminate_on_level_complete=False,
        episode_steps=10,
        max_episode_steps=12000,
        stagnation_steps=0,
    )

    assert terminated is True
    assert truncated is False
    assert reason == "death"


@pytest.mark.skipif(
    not Path("/Users/justinedwards/git/NES-SMB-RL/nes-py/nes_py/tests/games/super-mario-bros-1.nes").exists(),
    reason="requires the local nes-py bundled reference ROM",
)
def test_nes_smb_env_smoke():
    pytest.importorskip("nes_py")
    from rl_zoo3.nes_smb_env import NESMarioBrosEnv

    env = NESMarioBrosEnv(
        "/Users/justinedwards/git/NES-SMB-RL/nes-py/nes_py/tests/games/super-mario-bros-1.nes",
        render_mode=None,
        random_noop_max=0,
    )
    try:
        obs, info = env.reset()
        assert obs.shape == (80, 105, 1)
        assert obs.dtype == np.uint8
        assert info["world_display"] == 1
        assert info["level_display"] == 1

        obs, reward, terminated, truncated, info = env.step(4)
        assert obs.shape == (80, 105, 1)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "action_name" in info
        assert "level_progress" in info
        assert "furthest_progress" in info
    finally:
        env.close()
