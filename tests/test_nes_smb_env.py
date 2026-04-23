from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pytest

from rl_zoo3.nes_smb_env import (
    RewardConfig,
    NESMarioBrosEnv,
    action_tokens,
    build_metrics_from_ram,
    compute_reward_transition,
    defeated_enemy_count,
    direction_change_count,
    derive_termination_reason,
    furthest_progress_gain,
    hop_cluster_penalty,
    is_pipe_stall,
    jump_window_penalty,
    movement_trace_payload,
    nearest_enemy_ahead,
    oscillation_detected,
    prune_jump_timer_window,
    progress_window_gain,
)


def _make_ram() -> np.ndarray:
    return np.zeros(0x800, dtype=np.uint8)


def test_build_metrics_from_ram():
    ram = _make_ram()
    ram[0x001D] = 1
    ram[0x0057] = 28
    ram[0x006D] = 1
    ram[0x0086] = 32
    ram[0x00CE] = 191
    ram[0x0755] = 12
    ram[0x075F] = 0
    ram[0x075C] = 0
    ram[0x0770] = 1
    ram[0x07F8] = 3
    ram[0x07F9] = 9
    ram[0x07FA] = 7

    metrics = build_metrics_from_ram(ram)

    assert metrics.progress == 288
    assert metrics.level_progress == 288
    assert metrics.airborne is True
    assert metrics.x_speed == 28
    assert metrics.player_y_screen == 191
    assert metrics.world_display == 1
    assert metrics.level_display == 1
    assert metrics.timer == 397
    assert metrics.campaign_progress == 288


def test_movement_trace_payload():
    payload = movement_trace_payload(
        [
            {"progress": 600, "screen_x": 112, "x_speed": 48, "airborne": False},
            {"progress": 606, "screen_x": 112, "x_speed": 12, "airborne": False},
            {"progress": 606, "screen_x": 112, "x_speed": 0, "airborne": False},
        ]
    )

    assert payload["speed_trace"] == [48, 12, 0]
    assert payload["progress_trace"] == [600, 606, 606]
    assert payload["screen_x_trace"] == [112, 112, 112]
    assert payload["airborne_trace"] == [0, 0, 0]


def test_stagnation_event_includes_trace_fields():
    ram = _make_ram()
    ram[0x006D] = 0
    ram[0x0086] = 120
    ram[0x00CE] = 191
    ram[0x0755] = 112
    ram[0x0770] = 1
    metrics = build_metrics_from_ram(ram)

    env = NESMarioBrosEnv.__new__(NESMarioBrosEnv)
    env.reward_config = RewardConfig()
    env._event_log = deque(maxlen=32)
    env._event_seq = 0
    env._seen_event_keys = set()
    env._episode_steps = 77
    env._movement_trace = deque(
        [
            {"progress": 600, "screen_x": 112, "x_speed": 48, "airborne": False},
            {"progress": 606, "screen_x": 112, "x_speed": 12, "airborne": False},
            {"progress": 606, "screen_x": 112, "x_speed": 3, "airborne": False},
        ],
        maxlen=8,
    )

    env._maybe_record_behavior_events(
        metrics,
        nearest_enemy=None,
        pipe_stall_detected=False,
        stagnation_steps=3,
        stagnation_window_gain_value=6,
        furthest_window_gain_value=2,
        direction_changes=5,
        furthest_level_progress=180,
        hop_count=0,
        termination_reason=None,
    )

    event = env._event_log[-1]
    assert event["type"] == "stagnation_detected"
    assert event["stagnation_steps"] == 3
    assert event["stagnation_window_gain"] == 6
    assert event["furthest_window_gain"] == 2
    assert event["direction_changes"] == 5
    assert event["furthest_gap"] == 60
    assert event["speed_trace"] == [48, 12, 3]


def test_furthest_progress_gain_direction_changes_and_timer_prune():
    assert furthest_progress_gain([620, 620, 622, 622]) == 2
    assert direction_change_count([6, -4, 3, -2, 0, 5]) == 4

    jump_timers = deque([320, 315, 299])
    prune_jump_timer_window(jump_timers, current_timer=298, timer_window=20)

    assert list(jump_timers) == [315, 299]
    assert oscillation_detected(
        furthest_window_gain_value=2,
        direction_changes=5,
        furthest_gap=40,
        reward_config=RewardConfig(),
    )


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


def test_compute_reward_transition_powerup_bonus():
    ram_prev = _make_ram()
    ram_curr = _make_ram()

    ram_prev[0x0086] = 100
    ram_prev[0x0755] = 100
    ram_prev[0x075A] = 2
    ram_prev[0x0770] = 1
    ram_prev[0x000E] = 8
    ram_prev[0x000F] = 0

    ram_curr[:] = ram_prev
    ram_curr[0x000F] = 1

    previous = build_metrics_from_ram(ram_prev)
    current = build_metrics_from_ram(ram_curr)
    transition = compute_reward_transition(
        previous,
        current,
        RewardConfig(powerup_bonus=20.0),
        stagnation_steps=0,
    )

    assert transition.reward_parts["powerup"] == 20.0
    assert transition.reward > 0


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
    transition = compute_reward_transition(
        previous,
        current,
        RewardConfig(stagnation_steps=2, stagnation_min_progress=24),
        stagnation_steps=2,
        stagnation_window_ready=True,
        stagnation_window_gain_value=0,
        was_airborne=True,
    )

    assert transition.reward < 0
    assert transition.reward_parts["death_penalty"] == -25.0
    assert transition.reward_parts["airborne_death_penalty"] == -20.0
    assert "stagnation_penalty" in transition.reward_parts


def test_compute_reward_transition_penalizes_bad_jump_start_and_no_progress():
    ram_prev = _make_ram()
    ram_curr = _make_ram()

    ram_prev[0x0086] = 120
    ram_prev[0x0755] = 120
    ram_prev[0x075A] = 2
    ram_prev[0x0770] = 1
    ram_prev[0x000E] = 8
    ram_prev[0x0057] = 8

    ram_curr[:] = ram_prev
    ram_curr[0x0057] = 10

    previous = build_metrics_from_ram(ram_prev)
    current = build_metrics_from_ram(ram_curr)
    transition = compute_reward_transition(
        previous,
        current,
        RewardConfig(),
        stagnation_steps=0,
        jump_action_active=True,
        jump_started=True,
        grounded_run_steps=1,
        avg_x_speed=9.0,
        jump_no_progress_steps=3,
        is_moving_right=False,
        is_running_right=False,
    )

    assert transition.reward < 0
    assert transition.reward_parts["jump_start_penalty"] < 0
    assert transition.reward_parts["jump_no_progress_penalty"] < 0
    assert transition.reward_parts["jump_idle_penalty"] < 0


def test_compute_reward_transition_rewards_grounded_run():
    ram_prev = _make_ram()
    ram_curr = _make_ram()

    ram_prev[0x0086] = 120
    ram_prev[0x0755] = 120
    ram_prev[0x075A] = 2
    ram_prev[0x0770] = 1
    ram_prev[0x000E] = 8
    ram_prev[0x0057] = 24
    ram_prev[0x07F8] = 3

    ram_curr[:] = ram_prev
    ram_curr[0x0086] = 130
    ram_curr[0x0755] = 130
    ram_curr[0x0057] = 28

    previous = build_metrics_from_ram(ram_prev)
    current = build_metrics_from_ram(ram_curr)
    transition = compute_reward_transition(
        previous,
        current,
        RewardConfig(),
        stagnation_steps=0,
        grounded_run_steps=8,
        avg_x_speed=26.0,
        is_moving_right=True,
        is_running_right=True,
    )

    assert transition.reward_parts["run_ground_bonus"] > 0


def test_compute_reward_transition_rewards_full_speed_ground_run():
    ram_prev = _make_ram()
    ram_curr = _make_ram()

    ram_prev[0x0086] = 120
    ram_prev[0x0755] = 120
    ram_prev[0x075A] = 2
    ram_prev[0x0770] = 1
    ram_prev[0x000E] = 8
    ram_prev[0x0057] = 47
    ram_prev[0x07F8] = 3

    ram_curr[:] = ram_prev
    ram_curr[0x0086] = 126
    ram_curr[0x0755] = 126
    ram_curr[0x0057] = 48

    previous = build_metrics_from_ram(ram_prev)
    current = build_metrics_from_ram(ram_curr)
    transition = compute_reward_transition(
        previous,
        current,
        RewardConfig(full_speed_ground_bonus=0.12, full_speed_x_threshold=47, full_speed_run_steps=12),
        stagnation_steps=0,
        grounded_run_steps=12,
        avg_x_speed=47.5,
        is_moving_right=True,
        is_running_right=True,
    )

    assert transition.reward_parts["full_speed_ground_bonus"] == pytest.approx(0.12)


def test_compute_reward_transition_penalizes_short_jump_only_below_threshold():
    ram_prev = _make_ram()
    ram_curr = _make_ram()

    ram_prev[0x0086] = 120
    ram_prev[0x0755] = 120
    ram_prev[0x00CE] = 191
    ram_prev[0x075A] = 2
    ram_prev[0x0770] = 1
    ram_prev[0x000E] = 8

    ram_curr[:] = ram_prev

    previous = build_metrics_from_ram(ram_prev)
    current = build_metrics_from_ram(ram_curr)

    short_jump = compute_reward_transition(
        previous,
        current,
        RewardConfig(short_jump_penalty=0.06, short_jump_height_threshold=10),
        stagnation_steps=0,
        completed_jump_height=2,
    )
    tall_jump = compute_reward_transition(
        previous,
        current,
        RewardConfig(short_jump_penalty=0.06, short_jump_height_threshold=10),
        stagnation_steps=0,
        completed_jump_height=12,
    )

    assert short_jump.reward_parts["short_jump_penalty"] == pytest.approx(-0.048)
    assert "short_jump_penalty" not in tall_jump.reward_parts


def test_compute_reward_transition_rewards_good_jump_outcome():
    ram_prev = _make_ram()
    ram_curr = _make_ram()

    ram_prev[0x0086] = 120
    ram_prev[0x0755] = 120
    ram_prev[0x00CE] = 191
    ram_prev[0x075A] = 2
    ram_prev[0x0770] = 1
    ram_prev[0x000E] = 8

    ram_curr[:] = ram_prev
    ram_curr[0x0086] = 138

    previous = build_metrics_from_ram(ram_prev)
    current = build_metrics_from_ram(ram_curr)
    transition = compute_reward_transition(
        previous,
        current,
        RewardConfig(good_jump_bonus=0.12, good_jump_height_threshold=14, good_jump_progress_threshold=12),
        stagnation_steps=0,
        completed_jump_height=16,
        completed_jump_progress=18,
    )

    assert transition.reward_parts["good_jump_bonus"] == pytest.approx(0.12)


def test_compute_reward_transition_rewards_landing_spot_bonus():
    ram_prev = _make_ram()
    ram_curr = _make_ram()

    ram_prev[0x0086] = 120
    ram_prev[0x00CE] = 191
    ram_prev[0x075A] = 2
    ram_prev[0x0770] = 1
    ram_prev[0x000E] = 8

    ram_curr[:] = ram_prev
    ram_curr[0x0086] = 136
    ram_curr[0x00CE] = 176

    previous = build_metrics_from_ram(ram_prev)
    current = build_metrics_from_ram(ram_curr)
    transition = compute_reward_transition(
        previous,
        current,
        RewardConfig(landing_spot_bonus=0.08, landing_y_change_threshold=6, landing_progress_threshold=12),
        stagnation_steps=0,
        completed_jump_height=16,
        completed_jump_progress=16,
        completed_jump_landing_delta_y=15,
    )

    assert transition.reward_parts["landing_spot_bonus"] == pytest.approx(0.08)


def test_defeated_enemy_count_detects_enemy_defeat_transition():
    ram_prev = _make_ram()
    ram_curr = _make_ram()

    ram_prev[0x000F] = 1
    ram_prev[0x001E] = 0x06

    ram_curr[:] = ram_prev
    ram_curr[0x001E] = 0x09

    previous = build_metrics_from_ram(ram_prev)
    current = build_metrics_from_ram(ram_curr)

    assert defeated_enemy_count(previous, current) == 1


def test_compute_reward_transition_rewards_stomp_bonus():
    ram_prev = _make_ram()
    ram_curr = _make_ram()

    ram_prev[0x0086] = 120
    ram_prev[0x075A] = 2
    ram_prev[0x0770] = 1
    ram_prev[0x000E] = 8
    ram_prev[0x000F] = 1
    ram_prev[0x001E] = 0x06

    ram_curr[:] = ram_prev
    ram_curr[0x001E] = 0x09

    previous = build_metrics_from_ram(ram_prev)
    current = build_metrics_from_ram(ram_curr)
    transition = compute_reward_transition(previous, current, RewardConfig(stomp_bonus=3.0), stagnation_steps=0)

    assert transition.reward_parts["stomp_bonus"] == pytest.approx(3.0)


def test_nearest_enemy_ahead_detects_first_enemy():
    ram = _make_ram()
    ram[0x0086] = 80
    ram[0x0755] = 80
    ram[0x0770] = 1
    ram[0x000E] = 8
    ram[0x000F] = 1
    ram[0x0016] = 0x06
    ram[0x001E] = 0x06
    ram[0x006E] = 0
    ram[0x0087] = 104
    ram[0x00B6] = 1
    ram[0x00CF] = 191

    metrics = build_metrics_from_ram(ram)
    nearest = nearest_enemy_ahead(metrics)

    assert nearest is not None
    assert nearest["label"] == "goomba"
    assert nearest["dx"] == 24


def test_is_pipe_stall_detects_low_speed_wallup():
    ram = _make_ram()
    ram[0x0086] = 120
    ram[0x0755] = 112
    ram[0x0057] = 3
    ram[0x0770] = 1
    ram[0x000E] = 8

    metrics = build_metrics_from_ram(ram)

    assert is_pipe_stall(metrics, 8, RewardConfig(pipe_stall_screen_x_threshold=100, pipe_stall_speed_threshold=5))


def test_compute_reward_transition_penalizes_ignoring_critical_enemy():
    ram_prev = _make_ram()
    ram_curr = _make_ram()

    ram_prev[0x0086] = 100
    ram_prev[0x0755] = 100
    ram_prev[0x075A] = 2
    ram_prev[0x0770] = 1
    ram_prev[0x000E] = 8

    ram_curr[:] = ram_prev
    ram_curr[0x0086] = 102
    ram_curr[0x0755] = 102

    previous = build_metrics_from_ram(ram_prev)
    current = build_metrics_from_ram(ram_curr)
    transition = compute_reward_transition(
        previous,
        current,
        RewardConfig(enemy_ignore_penalty=0.16, enemy_immediate_dx_threshold=18),
        stagnation_steps=0,
        nearest_enemy_dx=12,
        jump_action_active=False,
    )

    assert transition.reward_parts["enemy_ignore_penalty"] == pytest.approx(-0.16)


def test_compute_reward_transition_penalizes_pipe_stall_choice():
    ram_prev = _make_ram()
    ram_curr = _make_ram()

    ram_prev[0x0086] = 120
    ram_prev[0x0755] = 112
    ram_prev[0x075A] = 2
    ram_prev[0x0770] = 1
    ram_prev[0x000E] = 8

    ram_curr[:] = ram_prev

    previous = build_metrics_from_ram(ram_prev)
    current = build_metrics_from_ram(ram_curr)
    transition = compute_reward_transition(
        previous,
        current,
        RewardConfig(pipe_stall_choice_penalty=0.2),
        stagnation_steps=0,
        pipe_stall_detected=True,
        jump_action_active=False,
    )

    assert transition.reward_parts["pipe_stall_choice_penalty"] == pytest.approx(-0.2)


def test_compute_reward_transition_penalizes_hop_spam_death():
    ram_prev = _make_ram()
    ram_curr = _make_ram()

    ram_prev[0x0086] = 120
    ram_prev[0x0755] = 112
    ram_prev[0x075A] = 2
    ram_prev[0x0770] = 1
    ram_prev[0x000E] = 8

    ram_curr[:] = ram_prev

    previous = build_metrics_from_ram(ram_prev)
    current = build_metrics_from_ram(ram_curr)
    transition = compute_reward_transition(
        previous,
        current,
        RewardConfig(death_penalty=25.0),
        stagnation_steps=0,
        hop_spam_death=True,
    )

    assert transition.reward_parts["hop_spam_death_penalty"] == pytest.approx(-25.0)


def test_progress_window_gain_and_jump_window_penalty():
    assert progress_window_gain([620, 624, 631, 629]) == 11
    assert progress_window_gain([]) == 0

    penalty = jump_window_penalty(
        jump_window_count=20,
        jump_penalty_threshold=12,
        jump_penalty_scale=0.02,
        jump_action_active=True,
        progress_delta=0,
    )
    assert penalty == pytest.approx(0.16)

    productive_penalty = jump_window_penalty(
        jump_window_count=20,
        jump_penalty_threshold=12,
        jump_penalty_scale=0.02,
        jump_action_active=True,
        progress_delta=8,
    )
    assert productive_penalty == pytest.approx(0.08)

    cluster_penalty = hop_cluster_penalty(
        jump_window_count=5,
        progress_window_gain_value=8,
        hop_cluster_count_threshold=4,
        hop_cluster_progress_threshold=24,
        hop_cluster_penalty_scale=0.16,
    )
    assert cluster_penalty == pytest.approx(0.5333333333)


def test_action_tokens_filters_macro_suffix():
    assert action_tokens("right_a_b_long") == ("right", "a", "b")


def test_derive_termination_reason():
    ram = _make_ram()
    ram[0x075A] = 1
    ram[0x000E] = 6
    ram[0x0770] = 1
    metrics = build_metrics_from_ram(ram)

    terminated, truncated, reason = derive_termination_reason(
        metrics,
        RewardConfig(stagnation_steps=240),
        hop_spam_death=False,
        level_complete=False,
        terminate_on_level_complete=False,
        episode_steps=10,
        max_episode_steps=12000,
        stagnation_steps=0,
    )

    assert terminated is True
    assert truncated is False
    assert reason == "death"


def test_derive_termination_reason_hop_spam_death():
    metrics = build_metrics_from_ram(_make_ram())

    terminated, truncated, reason = derive_termination_reason(
        metrics,
        RewardConfig(stagnation_steps=240),
        hop_spam_death=True,
        level_complete=False,
        terminate_on_level_complete=False,
        episode_steps=10,
        max_episode_steps=12000,
        stagnation_steps=0,
    )

    assert terminated is True
    assert truncated is False
    assert reason == "hop_spam_death"


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
        assert "stagnation_window_gain" in info
        assert "jump_window_count" in info
        assert env.action_space.n > 13
    finally:
        env.close()
