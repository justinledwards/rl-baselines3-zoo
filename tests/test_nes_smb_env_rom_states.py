from __future__ import annotations

"""ROM-grounded checks for SMB env assumptions.

These tests intentionally step a real nes-py ROM instead of writing synthetic RAM.
Synthetic tests prove pure reward math; this file proves that the RAM states our
reward math reads actually change the way we assume after concrete actions.
"""

import os
from pathlib import Path
from typing import Any

import pytest

from rl_zoo3.nes_smb_env import NESMarioBrosEnv

ROM_PATH = Path(
    os.environ.get(
        "NES_SMB_ROM_PATH",
        "/Users/justinedwards/git/NES-SMB-RL/nes-py/nes_py/tests/games/super-mario-bros-1.nes",
    )
)


def _rom_available() -> bool:
    return ROM_PATH.exists() and ROM_PATH.stat().st_size > 0


pytestmark = pytest.mark.skipif(not _rom_available(), reason=f"ROM not available at {ROM_PATH}")


def _make_env(**kwargs: Any) -> NESMarioBrosEnv:
    kwargs.setdefault("enable_hop_spam_death", False)
    return NESMarioBrosEnv(
        ROM_PATH,
        render_mode=None,
        random_noop_max=0,
        enable_left_jump_actions=False,
        debug_rewards=False,
        **kwargs,
    )


def _step(env: NESMarioBrosEnv, action_name: str) -> tuple[float, bool, bool, dict[str, Any]]:
    _, reward, terminated, truncated, info = env.step(env.action_names.index(action_name))
    return reward, terminated, truncated, info


def _event_types(info: dict[str, Any]) -> set[str]:
    return {event["type"] for event in info["events"]}


def test_rom_reset_starts_grounded_and_noop_stays_grounded():
    env = _make_env()
    try:
        _, info = env.reset()
        start_y = info["player_y_screen"]
        start_progress = info["level_progress"]

        assert info["airborne"] == 0
        assert info["x_speed"] == 0

        for _ in range(8):
            _reward, terminated, truncated, info = _step(env, "noop")
            assert not terminated
            assert not truncated
            assert info["airborne"] == 0
            assert info["player_y_screen"] == start_y
            assert info["level_progress"] == start_progress
    finally:
        env.close()


def test_rom_jump_action_sets_airborne_then_lands_grounded():
    env = _make_env()
    try:
        _, info = env.reset()
        ground_y = info["player_y_screen"]

        _reward, terminated, truncated, info = _step(env, "a")
        assert not terminated
        assert not truncated
        assert info["airborne"] == 1
        assert info["player_y_screen"] < ground_y
        assert "jump_start_penalty" in info["reward_parts"]
        assert "jump_idle_penalty" in info["reward_parts"]

        landed_info: dict[str, Any] | None = None
        for _ in range(24):
            _reward, terminated, truncated, info = _step(env, "noop")
            assert not terminated
            assert not truncated
            if info["airborne"] == 0:
                landed_info = info
                break

        assert landed_info is not None
        assert landed_info["player_y_screen"] == ground_y
        assert landed_info["airborne"] == 0
    finally:
        env.close()


def test_rom_grounded_running_reaches_full_speed_and_receives_ground_bonus():
    env = _make_env()
    try:
        _, info = env.reset()
        ground_y = info["player_y_screen"]
        saw_run_bonus = False
        saw_full_speed_bonus = False

        for _ in range(24):
            _reward, terminated, truncated, info = _step(env, "right_b")
            assert not terminated
            assert not truncated
            assert info["airborne"] == 0
            assert info["player_y_screen"] == ground_y
            saw_run_bonus = saw_run_bonus or "run_ground_bonus" in info["reward_parts"]
            saw_full_speed_bonus = saw_full_speed_bonus or "full_speed_ground_bonus" in info["reward_parts"]

        assert saw_run_bonus
        assert saw_full_speed_bonus
        assert info["x_speed"] >= env.reward_config.full_speed_x_threshold
    finally:
        env.close()


def test_rom_enemy_is_detected_by_level_position_and_critical_enemy_penalty_fires():
    env = _make_env()
    try:
        _, info = env.reset()
        first_enemy_info: dict[str, Any] | None = None
        critical_enemy_info: dict[str, Any] | None = None

        for _ in range(90):
            _reward, terminated, truncated, info = _step(env, "right")
            assert not truncated
            if first_enemy_info is None and "first_enemy_seen" in _event_types(info):
                first_enemy_info = info
            if "critical_enemy_seen" in _event_types(info):
                critical_enemy_info = info
                break
            if terminated:
                break

        assert first_enemy_info is not None
        assert first_enemy_info["nearest_enemy"] is not None
        assert first_enemy_info["nearest_enemy"]["level_x"] > first_enemy_info["level_progress"]
        assert first_enemy_info["nearest_enemy"]["dx"] > 0

        assert critical_enemy_info is not None
        assert critical_enemy_info["nearest_enemy"]["dx"] <= env.reward_config.enemy_immediate_dx_threshold
        assert critical_enemy_info["airborne"] == 0
        assert "enemy_ignore_penalty" in critical_enemy_info["reward_parts"]
    finally:
        env.close()


def test_rom_enemy_collision_terminates_with_death_near_enemy_event():
    env = _make_env()
    try:
        _, info = env.reset()

        for _ in range(90):
            _reward, terminated, truncated, info = _step(env, "right")
            assert not truncated
            if terminated:
                break

        assert terminated
        assert info["termination_reason"] == "death"
        assert "death_near_enemy" in _event_types(info)
    finally:
        env.close()


def test_rom_enemy_spawn_does_not_trigger_powerup_reward():
    env = _make_env()
    try:
        _, info = env.reset()

        for _ in range(70):
            _reward, terminated, truncated, info = _step(env, "right")
            assert not truncated
            assert "powerup" not in info["reward_parts"]
            if "first_enemy_seen" in _event_types(info):
                assert info["player_mode"] == 0
            if terminated:
                break
    finally:
        env.close()


def test_rom_coin_reward_appears_when_coin_counter_increments():
    env = _make_env()
    try:
        _, info = env.reset()
        coin_info: dict[str, Any] | None = None

        for action_name, count in (("right_b", 40), ("right_a_b_long", 4)):
            for _ in range(count):
                _reward, terminated, truncated, info = _step(env, action_name)
                assert not terminated
                assert not truncated
                if "coins" in info["reward_parts"]:
                    coin_info = info
                    break
            if coin_info is not None:
                break

        assert coin_info is not None
        assert coin_info["coins"] > 0
        assert coin_info["reward_parts"]["coins"] > 0
    finally:
        env.close()


def test_rom_ground_run_bonus_stops_when_pressed_against_pipe_without_new_furthest_progress():
    env = _make_env()
    try:
        _, info = env.reset()
        stalled_info: dict[str, Any] | None = None

        for action_name, count in (("right_b", 42), ("right_a_b_long", 5), ("right_b", 90)):
            for _ in range(count):
                _reward, terminated, truncated, info = _step(env, action_name)
                assert not terminated
                assert not truncated
                if info["level_progress"] >= 430 and info["new_furthest_delta"] == 0 and info["x_speed"] <= 5:
                    stalled_info = info
                    break
            if stalled_info is not None:
                break

        assert stalled_info is not None
        assert stalled_info["airborne"] == 0
        assert "progress" not in stalled_info["reward_parts"]
        assert "grounded_new_progress" not in stalled_info["reward_parts"]
        assert "run_ground_bonus" not in stalled_info["reward_parts"]
        assert "full_speed_ground_bonus" not in stalled_info["reward_parts"]
    finally:
        env.close()


def test_rom_pipe_stall_and_stagnation_events_are_logged_after_real_stall():
    env = _make_env()
    try:
        _, info = env.reset()
        pipe_info: dict[str, Any] | None = None
        stagnation_info: dict[str, Any] | None = None

        for action_name, count in (("right_b", 42), ("right_a_b_long", 5), ("right_b", 90)):
            for _ in range(count):
                _reward, terminated, truncated, info = _step(env, action_name)
                assert not terminated
                assert not truncated
                event_types = _event_types(info)
                if pipe_info is None and "pipe_stall" in event_types:
                    pipe_info = info
                if "stagnation_detected" in event_types:
                    stagnation_info = info
                    break
            if stagnation_info is not None:
                break

        assert pipe_info is not None
        assert pipe_info["pipe_stall_detected"] is True
        assert pipe_info["new_furthest_delta"] == 0
        assert "pipe_stall_choice_penalty" in pipe_info["reward_parts"]

        assert stagnation_info is not None
        assert stagnation_info["stagnation_steps"] >= env.reward_config.stagnation_event_steps
        assert "stagnation_detected" in _event_types(stagnation_info)
    finally:
        env.close()


def test_rom_hop_spam_death_can_terminate_when_enabled():
    env = _make_env(enable_hop_spam_death=True, hop_death_jump_threshold=2, hop_death_timer_window=400)
    try:
        _, info = env.reset()

        for _ in range(60):
            action_name = "a" if info["airborne"] == 0 else "noop"
            _reward, terminated, truncated, info = _step(env, action_name)
            if terminated or truncated:
                break

        assert terminated
        assert not truncated
        assert info["termination_reason"] == "hop_spam_death"
        assert "hop_spam_death" in _event_types(info)
    finally:
        env.close()


def test_rom_landing_on_higher_surface_is_grounded_not_airborne():
    env = _make_env()
    try:
        _, info = env.reset()
        ground_y = info["player_y_screen"]
        history: list[dict[str, Any]] = []

        for action_name, count in (
            ("right_b", 42),
            ("right_a_b_long", 5),
            ("right_b", 20),
            ("right_a_b_long", 5),
        ):
            for _ in range(count):
                _reward, terminated, truncated, info = _step(env, action_name)
                assert not terminated
                assert not truncated
                history.append(info)

        elevated_landings = [
            current
            for previous, current in zip(history, history[1:])
            if previous["airborne"] == 1
            and current["airborne"] == 0
            and current["player_y_screen"] < ground_y
            and current["level_progress"] > 430
        ]

        assert elevated_landings
        landing_info = elevated_landings[0]
        assert landing_info["airborne"] == 0
        assert landing_info["player_y_screen"] != ground_y
        assert "grounded_new_progress" in landing_info["reward_parts"]
    finally:
        env.close()
