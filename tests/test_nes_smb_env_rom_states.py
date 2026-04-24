from __future__ import annotations

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


def _make_env() -> NESMarioBrosEnv:
    return NESMarioBrosEnv(
        ROM_PATH,
        render_mode=None,
        random_noop_max=0,
        enable_left_jump_actions=False,
        enable_hop_spam_death=False,
        debug_rewards=False,
    )


def _step(env: NESMarioBrosEnv, action_name: str) -> tuple[float, bool, bool, dict[str, Any]]:
    _, reward, terminated, truncated, info = env.step(env.action_names.index(action_name))
    return reward, terminated, truncated, info


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
