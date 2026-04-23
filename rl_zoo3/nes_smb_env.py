from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

SCREEN_WIDTH = 256
PLAYING_GAME_MODE = 0x01
END_WORLD_GAME_MODE = 0x02
END_GAME_MODE = 0x03
FINAL_STAGE = (7, 3)
TRANSITION_PLAYER_STATES = (2, 3, 4, 5, 7)

BUTTON_NOOP = 0b00000000
BUTTON_RIGHT = 0b10000000
BUTTON_LEFT = 0b01000000
BUTTON_DOWN = 0b00100000
BUTTON_START = 0b00001000
BUTTON_B = 0b00000010
BUTTON_A = 0b00000001
BUTTON_TOKENS = frozenset({"right", "left", "down", "up", "a", "b", "noop"})

SLIM_ACTIONS: tuple[tuple[str, tuple[str, ...], int], ...] = (
    ("noop", ("NOOP",), BUTTON_NOOP),
    ("right", ("right",), BUTTON_RIGHT),
    ("left", ("left",), BUTTON_LEFT),
    ("down", ("down",), BUTTON_DOWN),
    ("right_b", ("right", "B"), BUTTON_RIGHT | BUTTON_B),
    ("left_b", ("left", "B"), BUTTON_LEFT | BUTTON_B),
    ("down_b", ("down", "B"), BUTTON_DOWN | BUTTON_B),
    ("a", ("A",), BUTTON_A),
    ("a_b", ("A", "B"), BUTTON_A | BUTTON_B),
    ("right_a", ("right", "A"), BUTTON_RIGHT | BUTTON_A),
    ("left_a", ("left", "A"), BUTTON_LEFT | BUTTON_A),
    ("right_a_b", ("right", "A", "B"), BUTTON_RIGHT | BUTTON_A | BUTTON_B),
    ("left_a_b", ("left", "A", "B"), BUTTON_LEFT | BUTTON_A | BUTTON_B),
)

LONG_JUMP_ACTIONS: tuple[tuple[str, tuple[str, ...], int], ...] = (
    ("right_a_long", ("right", "A"), BUTTON_RIGHT | BUTTON_A),
    ("right_a_b_long", ("right", "A", "B"), BUTTON_RIGHT | BUTTON_A | BUTTON_B),
    ("left_a_long", ("left", "A"), BUTTON_LEFT | BUTTON_A),
    ("left_a_b_long", ("left", "A", "B"), BUTTON_LEFT | BUTTON_A | BUTTON_B),
)


@dataclass(frozen=True)
class ActionSpec:
    name: str
    buttons: tuple[str, ...]
    sequence: tuple[int, ...]

    @property
    def uses_jump(self) -> bool:
        return any(mask & BUTTON_A for mask in self.sequence)


@dataclass(frozen=True)
class SMBRamAddresses:
    player_float_state: int = 0x001D
    player_x_speed: int = 0x0057
    player_page: int = 0x006D
    player_x: int = 0x0086
    player_y_screen: int = 0x00CE
    player_x_screen: int = 0x0755
    player_state: int = 0x000E
    player_mode: int = 0x000F
    enemy_drawn: tuple[int, ...] = (0x000F, 0x0010, 0x0011, 0x0012, 0x0013)
    enemy_types: tuple[int, ...] = (0x0016, 0x0017, 0x0018, 0x0019, 0x001A)
    enemy_states: tuple[int, ...] = (0x001E, 0x001F, 0x0020, 0x0021, 0x0022)
    enemy_pages: tuple[int, ...] = (0x006E, 0x006F, 0x0070, 0x0071, 0x0072)
    enemy_x_screens: tuple[int, ...] = (0x0087, 0x0088, 0x0089, 0x008A, 0x008B)
    enemy_y_viewports: tuple[int, ...] = (0x00B6, 0x00B7, 0x00B8, 0x00B9, 0x00BA)
    enemy_y_screens: tuple[int, ...] = (0x00CF, 0x00D0, 0x00D1, 0x00D2, 0x00D3)
    lives: int = 0x075A
    screen_scroll: int = 0x071C
    camera_x: int = 0x073F
    world: int = 0x075F
    level_dash: int = 0x075C
    game_mode: int = 0x0770
    timer_digit_1: int = 0x07F8
    timer_digit_2: int = 0x07F9
    timer_digit_3: int = 0x07FA
    coins: int = 0x075E
    area_offset: int = 0x0750


@dataclass(frozen=True)
class RewardConfig:
    progress_scale: float = 1.0 / 12.0
    time_penalty: float = 0.01
    death_penalty: float = 25.0
    airborne_death_penalty: float = 20.0
    flag_bonus: float = 500.0
    coin_bonus: float = 1.0
    powerup_bonus: float = 20.0
    stagnation_penalty: float = 1.0
    stagnation_steps: int = 240
    stagnation_window: int = 48
    stagnation_min_progress: int = 24
    death_states: tuple[int, ...] = (6, 11)
    time_progress_weight: float = 0.001
    jump_penalty_window: int = 32
    jump_penalty_threshold: int = 12
    jump_penalty_scale: float = 0.02
    jump_no_progress_penalty: float = 0.08
    jump_idle_penalty: float = 0.12
    jump_start_penalty: float = 0.3
    run_ground_bonus: float = 0.04
    full_speed_ground_bonus: float = 0.12
    full_speed_x_threshold: int = 47
    full_speed_run_steps: int = 12
    min_jump_speed: float = 24.0
    min_run_steps: int = 6
    short_jump_penalty: float = 0.06
    short_jump_height_threshold: int = 10
    good_jump_bonus: float = 0.12
    good_jump_height_threshold: int = 14
    good_jump_progress_threshold: int = 12
    landing_spot_bonus: float = 0.08
    landing_y_change_threshold: int = 6
    landing_progress_threshold: int = 12
    hop_cluster_penalty_scale: float = 0.16
    hop_cluster_count_threshold: int = 4
    hop_cluster_progress_threshold: int = 24
    stomp_bonus: float = 3.0
    enemy_ignore_penalty: float = 0.16
    enemy_immediate_dx_threshold: int = 18
    pipe_stall_choice_penalty: float = 0.2
    pipe_stall_screen_x_threshold: int = 100
    pipe_stall_speed_threshold: int = 5
    pipe_stall_progress_threshold: int = 12


@dataclass(frozen=True)
class SMBRamMetrics:
    float_state: int
    airborne: bool
    x_speed: int
    player_state: int
    player_mode: int
    player_page: int
    player_x: int
    player_y_screen: int
    player_x_screen: int
    screen_scroll: int
    camera_x: int
    lives: int
    world: int
    level: int
    area_offset: int
    game_mode: int
    coins: int
    timer: int
    progress: int
    level_progress: int
    campaign_progress: int
    enemy_drawn: tuple[int, ...]
    enemy_types: tuple[int, ...]
    enemy_states: tuple[int, ...]
    enemy_pages: tuple[int, ...]
    enemy_x_screens: tuple[int, ...]
    enemy_y_viewports: tuple[int, ...]
    enemy_y_screens: tuple[int, ...]

    @property
    def world_display(self) -> int:
        return self.world + 1

    @property
    def level_display(self) -> int:
        return self.level + 1

    def to_info(self) -> dict[str, int]:
        return {
            "world": self.world,
            "level": self.level,
            "world_display": self.world_display,
            "level_display": self.level_display,
            "player_state": self.player_state,
            "player_mode": self.player_mode,
            "float_state": self.float_state,
            "airborne": int(self.airborne),
            "x_speed": self.x_speed,
            "player_page": self.player_page,
            "player_x": self.player_x,
            "player_y_screen": self.player_y_screen,
            "player_x_screen": self.player_x_screen,
            "screen_scroll": self.screen_scroll,
            "camera_x": self.camera_x,
            "lives": self.lives,
            "area_offset": self.area_offset,
            "game_mode": self.game_mode,
            "coins": self.coins,
            "timer": self.timer,
            "progress": self.progress,
            "level_progress": self.level_progress,
            "campaign_progress": self.campaign_progress,
        }


@dataclass(frozen=True)
class RewardTransition:
    reward: float
    reward_parts: dict[str, float]
    stagnation_steps: int
    level_complete: bool
    jump_window_count: int
    stagnation_window_gain: int


def build_metrics_from_ram(ram: np.ndarray, addresses: SMBRamAddresses | None = None) -> SMBRamMetrics:
    addr = addresses or SMBRamAddresses()
    float_state = int(ram[addr.player_float_state])
    player_page = int(ram[addr.player_page])
    player_x = int(ram[addr.player_x])
    player_y_screen = int(ram[addr.player_y_screen])
    player_x_screen = int(ram[addr.player_x_screen])
    world = int(ram[addr.world])
    level = int(ram[addr.level_dash])
    progress = player_page * SCREEN_WIDTH + player_x
    level_progress = progress
    timer = int(ram[addr.timer_digit_1]) * 100 + int(ram[addr.timer_digit_2]) * 10 + int(ram[addr.timer_digit_3])
    campaign_progress = ((world * 4) + level) * 4096 + level_progress
    return SMBRamMetrics(
        float_state=float_state,
        airborne=float_state in (1, 2),
        x_speed=int(ram[addr.player_x_speed]),
        player_state=int(ram[addr.player_state]),
        player_mode=int(ram[addr.player_mode]),
        player_page=player_page,
        player_x=player_x,
        player_y_screen=player_y_screen,
        player_x_screen=player_x_screen,
        screen_scroll=int(ram[addr.screen_scroll]),
        camera_x=int(ram[addr.camera_x]),
        lives=int(ram[addr.lives]),
        world=world,
        level=level,
        area_offset=int(ram[addr.area_offset]),
        game_mode=int(ram[addr.game_mode]),
        coins=int(ram[addr.coins]),
        timer=timer,
        progress=progress,
        level_progress=level_progress,
        campaign_progress=campaign_progress,
        enemy_drawn=tuple(int(ram[offset]) for offset in addr.enemy_drawn),
        enemy_types=tuple(int(ram[offset]) for offset in addr.enemy_types),
        enemy_states=tuple(int(ram[offset]) for offset in addr.enemy_states),
        enemy_pages=tuple(int(ram[offset]) for offset in addr.enemy_pages),
        enemy_x_screens=tuple(int(ram[offset]) for offset in addr.enemy_x_screens),
        enemy_y_viewports=tuple(int(ram[offset]) for offset in addr.enemy_y_viewports),
        enemy_y_screens=tuple(int(ram[offset]) for offset in addr.enemy_y_screens),
    )


def is_dead(metrics: SMBRamMetrics, reward_config: RewardConfig | None = None) -> bool:
    config = reward_config or RewardConfig()
    return metrics.player_state in config.death_states


def is_transition_state(metrics: SMBRamMetrics) -> bool:
    return metrics.game_mode != PLAYING_GAME_MODE or metrics.player_state in TRANSITION_PLAYER_STATES


def is_game_complete(metrics: SMBRamMetrics) -> bool:
    return metrics.game_mode == END_GAME_MODE and (metrics.world, metrics.level) == FINAL_STAGE


def progress_window_gain(progress_window: Sequence[int]) -> int:
    if not progress_window:
        return 0
    return int(max(progress_window) - min(progress_window))


def movement_trace_payload(trace_window: Sequence[dict[str, int | bool]]) -> dict[str, list[int]]:
    samples = list(trace_window)
    return {
        "speed_trace": [int(sample["x_speed"]) for sample in samples],
        "progress_trace": [int(sample["progress"]) for sample in samples],
        "screen_x_trace": [int(sample["screen_x"]) for sample in samples],
        "airborne_trace": [1 if bool(sample["airborne"]) else 0 for sample in samples],
    }


def jump_window_penalty(
    *,
    jump_window_count: int,
    jump_penalty_threshold: int,
    jump_penalty_scale: float,
    jump_action_active: bool,
    progress_delta: int,
) -> float:
    if not jump_action_active or jump_window_count <= jump_penalty_threshold:
        return 0.0

    excess_jumps = jump_window_count - jump_penalty_threshold
    penalty = jump_penalty_scale * excess_jumps
    if progress_delta > 0:
        penalty *= 0.5
    return penalty


def hop_cluster_penalty(
    *,
    jump_window_count: int,
    progress_window_gain_value: int,
    hop_cluster_count_threshold: int,
    hop_cluster_progress_threshold: int,
    hop_cluster_penalty_scale: float,
) -> float:
    if jump_window_count < hop_cluster_count_threshold or progress_window_gain_value >= hop_cluster_progress_threshold:
        return 0.0

    excess_jumps = jump_window_count - hop_cluster_count_threshold + 1
    progress_shortfall = hop_cluster_progress_threshold - progress_window_gain_value
    return hop_cluster_penalty_scale * excess_jumps * (1.0 + progress_shortfall / max(1, hop_cluster_progress_threshold))


def enemy_type_label(enemy_type: int) -> str:
    mapping = {
        0x00: "koopa",
        0x01: "koopa",
        0x03: "koopa",
        0x04: "koopa",
        0x06: "goomba",
        0x09: "koopa",
        0x0D: "piranha_plant",
        0x0E: "koopa",
        0x11: "lakitu",
        0x12: "spiny_egg",
        0x24: "lift",
        0x25: "lift",
        0x26: "lift",
        0x27: "lift",
        0x28: "lift",
        0x29: "lift",
        0x2A: "lift",
        0x2B: "lift",
        0x2C: "lift",
        0x2D: "bowser",
        0x32: "spring",
        0x37: "goomba",
        0x38: "goomba",
        0x3B: "koopa",
        0x3C: "koopa",
    }
    return mapping.get(enemy_type, "unknown")


def nearest_enemy_ahead(metrics: SMBRamMetrics) -> dict[str, int | str] | None:
    nearest: dict[str, int | str] | None = None
    for slot, (drawn, enemy_type, enemy_state, page, x_screen, y_viewport, y_screen) in enumerate(
        zip(
            metrics.enemy_drawn,
            metrics.enemy_types,
            metrics.enemy_states,
            metrics.enemy_pages,
            metrics.enemy_x_screens,
            metrics.enemy_y_viewports,
            metrics.enemy_y_screens,
            strict=True,
        )
    ):
        label = enemy_type_label(enemy_type)
        on_screen = drawn != 0 and label != "unknown" and y_viewport == 1 and 0 < x_screen < 255
        if not on_screen:
            continue
        dx = x_screen - metrics.player_x_screen
        if dx <= 0:
            continue
        candidate = {
            "slot": slot,
            "label": label,
            "dx": dx,
            "enemy_type": enemy_type,
            "enemy_state": enemy_state,
            "page": page,
            "x_screen": x_screen,
            "y_screen": y_screen,
        }
        if nearest is None or int(candidate["dx"]) < int(nearest["dx"]):
            nearest = candidate
    return nearest


def is_pipe_stall(metrics: SMBRamMetrics, progress_window_gain_value: int, reward_config: RewardConfig) -> bool:
    return (
        not metrics.airborne
        and metrics.player_x_screen >= reward_config.pipe_stall_screen_x_threshold
        and metrics.x_speed < reward_config.pipe_stall_speed_threshold
        and progress_window_gain_value < reward_config.pipe_stall_progress_threshold
    )


def defeated_enemy_count(previous: SMBRamMetrics, current: SMBRamMetrics) -> int:
    defeated_states = {0x09, 0x0B, 0x1F}
    defeats = 0
    for prev_drawn, prev_state, curr_state in zip(
        previous.enemy_drawn, previous.enemy_states, current.enemy_states, strict=True
    ):
        if prev_drawn and prev_state not in defeated_states and curr_state in defeated_states:
            defeats += 1
    return defeats


def behavior_reward_parts(
    *,
    current: SMBRamMetrics,
    reward_config: RewardConfig,
    progress_delta: int,
    jump_action_active: bool,
    jump_started: bool,
    was_airborne: bool,
    grounded_run_steps: int,
    avg_x_speed: float,
    jump_no_progress_steps: int,
    is_moving_right: bool,
    is_running_right: bool,
    completed_jump_height: int | None,
    completed_jump_progress: int | None,
    completed_jump_landing_delta_y: int | None,
    nearest_enemy_dx: int | None,
    pipe_stall_detected: bool,
) -> dict[str, float]:
    reward_parts: dict[str, float] = {}

    grounded_run = is_running_right and not jump_action_active and not current.airborne
    if grounded_run and avg_x_speed > 12.0:
        streak_scale = min(1.0 + 0.15 * max(0, grounded_run_steps - 1), 2.5)
        grounded_run_bonus = streak_scale * (reward_config.run_ground_bonus + max(0.0, avg_x_speed - 16.0) * 0.004)
        reward_parts["run_ground_bonus"] = grounded_run_bonus
        if (
            current.x_speed >= reward_config.full_speed_x_threshold
            and grounded_run_steps >= reward_config.full_speed_run_steps
        ):
            reward_parts["full_speed_ground_bonus"] = reward_config.full_speed_ground_bonus

    if jump_started and not was_airborne:
        speed_deficit = max(0.0, reward_config.min_jump_speed - avg_x_speed)
        run_deficit = max(0, reward_config.min_run_steps - grounded_run_steps)
        if speed_deficit > 0 or run_deficit > 0 or not is_running_right:
            jump_start_penalty = reward_config.jump_start_penalty * (
                1.0
                + speed_deficit / max(1.0, reward_config.min_jump_speed)
                + run_deficit / max(1, reward_config.min_run_steps)
                + (0.5 if not is_running_right else 0.0)
            )
            reward_parts["jump_start_penalty"] = -jump_start_penalty

    if completed_jump_height is not None and completed_jump_height < reward_config.short_jump_height_threshold:
        short_jump_penalty = reward_config.short_jump_penalty * (
            1.0 - (completed_jump_height / max(1, reward_config.short_jump_height_threshold))
        )
        reward_parts["short_jump_penalty"] = -short_jump_penalty

    if (
        completed_jump_height is not None
        and completed_jump_progress is not None
        and completed_jump_height >= reward_config.good_jump_height_threshold
        and completed_jump_progress >= reward_config.good_jump_progress_threshold
    ):
        reward_parts["good_jump_bonus"] = reward_config.good_jump_bonus

    if (
        completed_jump_landing_delta_y is not None
        and completed_jump_progress is not None
        and completed_jump_landing_delta_y >= reward_config.landing_y_change_threshold
        and completed_jump_progress >= reward_config.landing_progress_threshold
    ):
        reward_parts["landing_spot_bonus"] = reward_config.landing_spot_bonus

    if jump_action_active and progress_delta <= 0:
        jump_no_progress_penalty = reward_config.jump_no_progress_penalty * (
            (1 + jump_no_progress_steps) if progress_delta == 0 else (2 + jump_no_progress_steps)
        )
        reward_parts["jump_no_progress_penalty"] = -jump_no_progress_penalty

        if avg_x_speed < 16.0 or not is_moving_right:
            reward_parts["jump_idle_penalty"] = -reward_config.jump_idle_penalty

    if nearest_enemy_dx is not None and nearest_enemy_dx <= reward_config.enemy_immediate_dx_threshold:
        if not jump_action_active and not current.airborne:
            reward_parts["enemy_ignore_penalty"] = -reward_config.enemy_ignore_penalty

    if pipe_stall_detected and not jump_action_active:
        reward_parts["pipe_stall_choice_penalty"] = -reward_config.pipe_stall_choice_penalty

    return reward_parts


def compute_reward_transition(
    previous: SMBRamMetrics,
    current: SMBRamMetrics,
    reward_config: RewardConfig,
    stagnation_steps: int,
    *,
    stagnation_window_ready: bool = False,
    stagnation_window_gain_value: int = 0,
    jump_window_count: int = 0,
    jump_action_active: bool = False,
    jump_started: bool = False,
    was_airborne: bool = False,
    grounded_run_steps: int = 0,
    avg_x_speed: float = 0.0,
    jump_no_progress_steps: int = 0,
    is_moving_right: bool = False,
    is_running_right: bool = False,
    completed_jump_height: int | None = None,
    completed_jump_progress: int | None = None,
    completed_jump_landing_delta_y: int | None = None,
    nearest_enemy_dx: int | None = None,
    pipe_stall_detected: bool = False,
) -> RewardTransition:
    reward = 0.0
    reward_parts: dict[str, float] = {}
    zone_changed = (current.world, current.level, current.area_offset) != (
        previous.world,
        previous.level,
        previous.area_offset,
    )
    level_changed = (current.world, current.level) != (previous.world, previous.level)
    progress_delta = current.progress - previous.progress

    if progress_delta > 0:
        base_progress_reward = progress_delta * reward_config.progress_scale
        time_bonus = base_progress_reward * (current.timer * reward_config.time_progress_weight)
        total_progress_reward = base_progress_reward + time_bonus
        reward += total_progress_reward
        reward_parts["progress"] = total_progress_reward

    if is_transition_state(current) or zone_changed:
        next_stagnation_steps = 0
    elif stagnation_window_ready:
        next_stagnation_steps = (
            stagnation_steps + 1 if stagnation_window_gain_value < reward_config.stagnation_min_progress else 0
        )
    elif progress_delta != 0:
        next_stagnation_steps = 0
    else:
        next_stagnation_steps = stagnation_steps + 1

    reward -= reward_config.time_penalty
    reward_parts["time_penalty"] = -reward_config.time_penalty

    if current.lives < previous.lives:
        death_delta = previous.lives - current.lives
        death_penalty = reward_config.death_penalty * death_delta
        reward -= death_penalty
        reward_parts["death_penalty"] = -death_penalty
        if was_airborne:
            airborne_penalty = reward_config.airborne_death_penalty * death_delta
            reward -= airborne_penalty
            reward_parts["airborne_death_penalty"] = -airborne_penalty

    if level_changed:
        reward += reward_config.flag_bonus
        reward_parts["level_complete"] = reward_config.flag_bonus

    coin_delta = current.coins - previous.coins
    if coin_delta > 0:
        coin_reward = coin_delta * reward_config.coin_bonus
        reward += coin_reward
        reward_parts["coins"] = coin_reward

    player_mode_delta = current.player_mode - previous.player_mode
    if player_mode_delta > 0:
        powerup_reward = player_mode_delta * reward_config.powerup_bonus
        reward += powerup_reward
        reward_parts["powerup"] = powerup_reward

    reward_parts.update(
        behavior_reward_parts(
            current=current,
            reward_config=reward_config,
            progress_delta=progress_delta,
            jump_action_active=jump_action_active,
            jump_started=jump_started,
            was_airborne=was_airborne,
            grounded_run_steps=grounded_run_steps,
            avg_x_speed=avg_x_speed,
            jump_no_progress_steps=jump_no_progress_steps,
            is_moving_right=is_moving_right,
            is_running_right=is_running_right,
            completed_jump_height=completed_jump_height,
            completed_jump_progress=completed_jump_progress,
            completed_jump_landing_delta_y=completed_jump_landing_delta_y,
            nearest_enemy_dx=nearest_enemy_dx,
            pipe_stall_detected=pipe_stall_detected,
        )
    )
    reward += sum(
        reward_parts[name]
        for name in reward_parts
        if name
        in {
            "run_ground_bonus",
            "full_speed_ground_bonus",
            "jump_start_penalty",
            "short_jump_penalty",
            "good_jump_bonus",
            "landing_spot_bonus",
            "jump_no_progress_penalty",
            "jump_idle_penalty",
            "enemy_ignore_penalty",
            "pipe_stall_choice_penalty",
        }
    )

    stomp_count = defeated_enemy_count(previous, current)
    if stomp_count > 0:
        stomp_bonus = stomp_count * reward_config.stomp_bonus
        reward += stomp_bonus
        reward_parts["stomp_bonus"] = stomp_bonus

    excessive_jump_penalty = jump_window_penalty(
        jump_window_count=jump_window_count,
        jump_penalty_threshold=reward_config.jump_penalty_threshold,
        jump_penalty_scale=reward_config.jump_penalty_scale,
        jump_action_active=jump_action_active,
        progress_delta=progress_delta,
    )
    if excessive_jump_penalty > 0:
        reward -= excessive_jump_penalty
        reward_parts["jump_window_penalty"] = -excessive_jump_penalty

    clustered_hop_penalty = hop_cluster_penalty(
        jump_window_count=jump_window_count,
        progress_window_gain_value=stagnation_window_gain_value,
        hop_cluster_count_threshold=reward_config.hop_cluster_count_threshold,
        hop_cluster_progress_threshold=reward_config.hop_cluster_progress_threshold,
        hop_cluster_penalty_scale=reward_config.hop_cluster_penalty_scale,
    )
    if clustered_hop_penalty > 0:
        reward -= clustered_hop_penalty
        reward_parts["hop_cluster_penalty"] = -clustered_hop_penalty

    if not zone_changed and next_stagnation_steps >= reward_config.stagnation_steps:
        steps_over = next_stagnation_steps - reward_config.stagnation_steps
        multiplier = 1.0 + (steps_over * 0.1) ** 1.5
        stagnation_penalty = reward_config.stagnation_penalty * multiplier
        reward -= stagnation_penalty
        reward_parts["stagnation_penalty"] = -stagnation_penalty

    return RewardTransition(
        reward=reward,
        reward_parts=reward_parts,
        stagnation_steps=next_stagnation_steps,
        level_complete=level_changed,
        jump_window_count=jump_window_count,
        stagnation_window_gain=stagnation_window_gain_value,
    )


def derive_termination_reason(
    metrics: SMBRamMetrics,
    reward_config: RewardConfig,
    *,
    level_complete: bool,
    terminate_on_level_complete: bool,
    episode_steps: int,
    max_episode_steps: int,
    stagnation_steps: int,
) -> tuple[bool, bool, str | None]:
    game_complete = is_game_complete(metrics)
    dead = is_dead(metrics, reward_config)
    hit_episode_limit = episode_steps >= max_episode_steps
    hit_stagnation = stagnation_steps >= reward_config.stagnation_steps

    terminated = game_complete or dead or (terminate_on_level_complete and level_complete)
    truncated = not terminated and (hit_episode_limit or hit_stagnation)

    if game_complete:
        reason = "game_complete"
    elif dead and hit_stagnation:
        reason = "death_after_stagnation"
    elif dead and hit_episode_limit:
        reason = "death_at_episode_limit"
    elif dead:
        reason = "death"
    elif terminate_on_level_complete and level_complete:
        reason = "level_complete"
    elif hit_episode_limit:
        reason = "max_episode_steps"
    elif hit_stagnation:
        reason = "stagnation"
    else:
        reason = None

    return terminated, truncated, reason


def action_tokens(action_name: str) -> tuple[str, ...]:
    if action_name == "noop":
        return ()
    return tuple(token for token in (part.lower() for part in action_name.split("_")) if token in BUTTON_TOKENS)


class NESMarioBrosEnv(gym.Env):
    metadata: ClassVar[dict[str, list[str] | int]] = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str | Path,
        *,
        render_mode: str | None = None,
        frames_per_action: int = 2,
        resize_shape: tuple[int, int] = (105, 80),
        grayscale: bool = True,
        max_episode_steps: int = 12000,
        stagnation_steps: int = 240,
        stagnation_window: int = 48,
        stagnation_min_progress: int = 24,
        terminate_on_level_complete: bool = False,
        random_noop_max: int = 30,
        jump_penalty_window: int = 32,
        jump_penalty_threshold: int = 12,
        jump_penalty_scale: float = 0.02,
        jump_no_progress_penalty: float = 0.08,
        jump_idle_penalty: float = 0.12,
        jump_start_penalty: float = 0.3,
        run_ground_bonus: float = 0.04,
        full_speed_ground_bonus: float = 0.12,
        full_speed_x_threshold: int = 47,
        full_speed_run_steps: int = 12,
        min_jump_speed: float = 24.0,
        min_run_steps: int = 6,
        short_jump_penalty: float = 0.06,
        short_jump_height_threshold: int = 10,
        good_jump_bonus: float = 0.12,
        good_jump_height_threshold: int = 14,
        good_jump_progress_threshold: int = 12,
        landing_spot_bonus: float = 0.08,
        landing_y_change_threshold: int = 6,
        landing_progress_threshold: int = 12,
        hop_cluster_penalty_scale: float = 0.16,
        hop_cluster_count_threshold: int = 4,
        hop_cluster_progress_threshold: int = 24,
        stomp_bonus: float = 3.0,
        enemy_ignore_penalty: float = 0.16,
        enemy_immediate_dx_threshold: int = 18,
        pipe_stall_choice_penalty: float = 0.2,
        pipe_stall_screen_x_threshold: int = 100,
        pipe_stall_speed_threshold: int = 5,
        pipe_stall_progress_threshold: int = 12,
        enable_long_jumps: bool = True,
        macro_jump_hold_frames: int = 6,
        macro_jump_release_frames: int = 1,
        settle_frames: int = 120,
        start_press_frames: int = 10,
        start_wait_frames: int = 180,
    ):
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        super().__init__()
        self.rom_path = Path(rom_path).expanduser().resolve()
        self.render_mode = render_mode
        self.frames_per_action = frames_per_action
        self.resize_shape = resize_shape
        self.grayscale = grayscale
        self.max_episode_steps = max_episode_steps
        self.terminate_on_level_complete = terminate_on_level_complete
        self.random_noop_max = random_noop_max
        self.settle_frames = settle_frames
        self.start_press_frames = start_press_frames
        self.start_wait_frames = start_wait_frames
        self.enable_long_jumps = enable_long_jumps
        self.macro_jump_hold_frames = macro_jump_hold_frames
        self.macro_jump_release_frames = macro_jump_release_frames
        self.reward_config = RewardConfig(
            stagnation_steps=stagnation_steps,
            stagnation_window=stagnation_window,
            stagnation_min_progress=stagnation_min_progress,
            jump_penalty_window=jump_penalty_window,
            jump_penalty_threshold=jump_penalty_threshold,
            jump_penalty_scale=jump_penalty_scale,
            jump_no_progress_penalty=jump_no_progress_penalty,
            jump_idle_penalty=jump_idle_penalty,
            jump_start_penalty=jump_start_penalty,
            run_ground_bonus=run_ground_bonus,
            full_speed_ground_bonus=full_speed_ground_bonus,
            full_speed_x_threshold=full_speed_x_threshold,
            full_speed_run_steps=full_speed_run_steps,
            min_jump_speed=min_jump_speed,
            min_run_steps=min_run_steps,
            short_jump_penalty=short_jump_penalty,
            short_jump_height_threshold=short_jump_height_threshold,
            good_jump_bonus=good_jump_bonus,
            good_jump_height_threshold=good_jump_height_threshold,
            good_jump_progress_threshold=good_jump_progress_threshold,
            landing_spot_bonus=landing_spot_bonus,
            landing_y_change_threshold=landing_y_change_threshold,
            landing_progress_threshold=landing_progress_threshold,
            hop_cluster_penalty_scale=hop_cluster_penalty_scale,
            hop_cluster_count_threshold=hop_cluster_count_threshold,
            hop_cluster_progress_threshold=hop_cluster_progress_threshold,
            stomp_bonus=stomp_bonus,
            enemy_ignore_penalty=enemy_ignore_penalty,
            enemy_immediate_dx_threshold=enemy_immediate_dx_threshold,
            pipe_stall_choice_penalty=pipe_stall_choice_penalty,
            pipe_stall_screen_x_threshold=pipe_stall_screen_x_threshold,
            pipe_stall_speed_threshold=pipe_stall_speed_threshold,
            pipe_stall_progress_threshold=pipe_stall_progress_threshold,
        )
        self._addresses = SMBRamAddresses()

        height, width = resize_shape[1], resize_shape[0]
        channels = 1 if grayscale else 3
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, channels), dtype=np.uint8)
        self._action_specs = self._build_action_specs()
        self.action_space = spaces.Discrete(len(self._action_specs))

        self._base_env = None
        self._last_frame: np.ndarray | None = None
        self._last_metrics: SMBRamMetrics | None = None
        self._episode_steps = 0
        self._stagnation_steps = 0
        self._furthest_campaign_progress = 0
        self._furthest_level_progress = 0
        self._progress_window: deque[int] = deque(maxlen=self.reward_config.stagnation_window)
        self._jump_window: deque[int] = deque(maxlen=self.reward_config.jump_penalty_window)
        self._speed_window: deque[int] = deque(maxlen=8)
        self._movement_trace: deque[dict[str, int | bool]] = deque(maxlen=8)
        self._prev_jump_action_active = False
        self._ground_run_steps = 0
        self._jump_no_progress_steps = 0
        self._jump_start_y: int | None = None
        self._jump_peak_y: int | None = None
        self._jump_start_progress: int | None = None
        self._last_completed_jump_height = 0
        self._last_completed_jump_progress = 0
        self._last_completed_jump_landing_delta_y = 0
        self._event_log: deque[dict[str, Any]] = deque(maxlen=32)
        self._event_seq = 0
        self._seen_event_keys: set[str] = set()

        self._load_backend()
        self._boot_to_level_start()

    def _build_action_specs(self) -> tuple[ActionSpec, ...]:
        specs = [self._make_action_spec(name, buttons, mask) for name, buttons, mask in SLIM_ACTIONS]
        if self.enable_long_jumps:
            specs.extend(
                self._make_action_spec(name, buttons, mask, long_jump=True) for name, buttons, mask in LONG_JUMP_ACTIONS
            )
        return tuple(specs)

    def _make_action_spec(
        self,
        name: str,
        buttons: tuple[str, ...],
        mask: int,
        *,
        long_jump: bool = False,
    ) -> ActionSpec:
        if long_jump and (mask & BUTTON_A):
            hold_frames = max(self.frames_per_action, self.macro_jump_hold_frames)
            release_frames = max(self.macro_jump_release_frames, 0)
            release_mask = mask & ~BUTTON_A
            sequence = (mask,) * hold_frames + (release_mask,) * release_frames
        else:
            sequence = (mask,) * max(self.frames_per_action, 1)
        return ActionSpec(name=name, buttons=buttons, sequence=sequence)

    def _load_backend(self) -> None:
        try:
            from nes_py import NESEnv
        except ImportError as exc:
            raise ImportError(
                "NES-SMB-v0 requires nes-py in the active environment. "
                "Run `mise run setup` in rl-baselines3-zoo to install the sibling checkout."
            ) from exc

        self._base_env = NESEnv(str(self.rom_path))

    def _boot_to_level_start(self) -> None:
        assert self._base_env is not None
        self._base_env.reset()
        for _ in range(self.settle_frames):
            self._base_env.step(BUTTON_NOOP)
        for _ in range(self.start_press_frames):
            self._base_env.step(BUTTON_START)
        for _ in range(self.start_wait_frames):
            self._base_env.step(BUTTON_NOOP)
        self._base_env._backup()

    def _metrics(self) -> SMBRamMetrics:
        assert self._base_env is not None
        return build_metrics_from_ram(self._base_env.ram, self._addresses)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, self.resize_shape, interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = frame[:, :, None]
        return frame.astype(np.uint8)

    def _render_human_if_needed(self) -> None:
        if self.render_mode == "human":
            assert self._base_env is not None
            self._base_env.render("human")

    def _build_info(
        self,
        metrics: SMBRamMetrics,
        *,
        action_name: str,
        reward_parts: dict[str, float],
        termination_reason: str | None,
        jump_window_count: int = 0,
        stagnation_window_gain_value: int = 0,
        avg_x_speed: float = 0.0,
        current_jump_height: int = 0,
        current_jump_progress: int = 0,
        nearest_enemy: dict[str, int | str] | None = None,
        pipe_stall_detected: bool = False,
    ) -> dict[str, Any]:
        info = metrics.to_info()
        info.update(
            {
                "action_name": action_name,
                "action_tokens": action_tokens(action_name),
                "episode_steps": self._episode_steps,
                "furthest_progress": self._furthest_campaign_progress,
                "furthest_level_progress": self._furthest_level_progress,
                "stagnation_steps": self._stagnation_steps,
                "stagnation_window_gain": stagnation_window_gain_value,
                "termination_reason": termination_reason,
                "jump_window_count": jump_window_count,
                "ground_run_steps": self._ground_run_steps,
                "jump_no_progress_steps": self._jump_no_progress_steps,
                "avg_x_speed": avg_x_speed,
                "current_jump_height": current_jump_height,
                "current_jump_progress": current_jump_progress,
                "last_completed_jump_height": self._last_completed_jump_height,
                "last_completed_jump_progress": self._last_completed_jump_progress,
                "last_completed_jump_landing_delta_y": self._last_completed_jump_landing_delta_y,
                "nearest_enemy": nearest_enemy,
                "pipe_stall_detected": pipe_stall_detected,
                "last_event": self._event_log[-1] if self._event_log else None,
                "reward_parts": reward_parts,
                "render_mode": self.render_mode,
            }
        )
        return info

    def _append_event(self, event_type: str, metrics: SMBRamMetrics, **payload: Any) -> None:
        self._event_seq += 1
        event = {
            "id": self._event_seq,
            "type": event_type,
            "episode_steps": self._episode_steps,
            "progress": metrics.progress,
            "level_progress": metrics.level_progress,
            "world_display": metrics.world_display,
            "level_display": metrics.level_display,
            "area_offset": metrics.area_offset,
            "player_x_screen": metrics.player_x_screen,
            "player_y_screen": metrics.player_y_screen,
            **payload,
        }
        self._event_log.append(event)

    @staticmethod
    def _trace_sample(metrics: SMBRamMetrics) -> dict[str, int | bool]:
        return {
            "progress": metrics.progress,
            "screen_x": metrics.player_x_screen,
            "x_speed": metrics.x_speed,
            "airborne": metrics.airborne,
        }

    def _maybe_record_behavior_events(
        self,
        metrics: SMBRamMetrics,
        *,
        nearest_enemy: dict[str, int | str] | None,
        pipe_stall_detected: bool,
        stagnation_steps: int,
        stagnation_window_gain_value: int,
        furthest_level_progress: int,
        termination_reason: str | None,
    ) -> None:
        if nearest_enemy is not None and "first_enemy_seen" not in self._seen_event_keys:
            self._append_event(
                "first_enemy_seen",
                metrics,
                enemy_label=nearest_enemy["label"],
                enemy_dx=nearest_enemy["dx"],
                enemy_slot=nearest_enemy["slot"],
            )
            self._seen_event_keys.add("first_enemy_seen")

        if nearest_enemy is not None and int(nearest_enemy["dx"]) <= self.reward_config.enemy_immediate_dx_threshold:
            event_key = "critical_enemy_seen"
            if event_key not in self._seen_event_keys:
                self._append_event(
                    "critical_enemy_seen",
                    metrics,
                    enemy_label=nearest_enemy["label"],
                    enemy_dx=nearest_enemy["dx"],
                    enemy_slot=nearest_enemy["slot"],
                )
                self._seen_event_keys.add(event_key)

        if pipe_stall_detected and "pipe_stall" not in self._seen_event_keys:
            self._append_event("pipe_stall", metrics, x_speed=metrics.x_speed, **movement_trace_payload(self._movement_trace))
            self._seen_event_keys.add("pipe_stall")

        if stagnation_steps > 0 and "stagnation_detected" not in self._seen_event_keys:
            self._append_event(
                "stagnation_detected",
                metrics,
                x_speed=metrics.x_speed,
                stagnation_steps=stagnation_steps,
                stagnation_window_gain=stagnation_window_gain_value,
                furthest_gap=max(0, furthest_level_progress - metrics.level_progress),
                **movement_trace_payload(self._movement_trace),
            )
            self._seen_event_keys.add("stagnation_detected")

        if termination_reason == "death" and nearest_enemy is not None:
            self._append_event(
                "death_near_enemy",
                metrics,
                enemy_label=nearest_enemy["label"],
                enemy_dx=nearest_enemy["dx"],
            )
        elif termination_reason == "stagnation" and pipe_stall_detected:
            self._append_event(
                "stagnation_at_pipe",
                metrics,
                x_speed=metrics.x_speed,
                **movement_trace_payload(self._movement_trace),
            )

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._action_specs)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        assert self._base_env is not None

        frame, _info = self._base_env.reset(seed=seed)
        self._episode_steps = 0
        self._stagnation_steps = 0

        if self.random_noop_max > 0:
            noop_steps = int(self.np_random.integers(0, self.random_noop_max + 1))
            for _ in range(noop_steps):
                frame, _reward, _terminated, _truncated, _info = self._base_env.step(BUTTON_NOOP)

        metrics = self._metrics()
        self._last_metrics = metrics
        self._furthest_campaign_progress = metrics.campaign_progress
        self._furthest_level_progress = metrics.level_progress
        self._progress_window.clear()
        self._progress_window.append(metrics.level_progress)
        self._jump_window.clear()
        self._speed_window.clear()
        self._speed_window.append(metrics.x_speed)
        self._movement_trace.clear()
        self._movement_trace.append(self._trace_sample(metrics))
        self._prev_jump_action_active = False
        self._ground_run_steps = 0
        self._jump_no_progress_steps = 0
        self._jump_start_y = None
        self._jump_peak_y = None
        self._jump_start_progress = None
        self._last_completed_jump_height = 0
        self._last_completed_jump_progress = 0
        self._last_completed_jump_landing_delta_y = 0
        self._event_log.clear()
        self._event_seq = 0
        self._seen_event_keys.clear()
        self._last_frame = np.array(frame, copy=True)

        self._render_human_if_needed()
        return self._preprocess(self._last_frame), self._build_info(
            metrics,
            action_name="noop",
            reward_parts={},
            termination_reason=None,
            jump_window_count=0,
            stagnation_window_gain_value=0,
            avg_x_speed=float(metrics.x_speed),
            current_jump_height=0,
            current_jump_progress=0,
            nearest_enemy=nearest_enemy_ahead(metrics),
            pipe_stall_detected=False,
        )

    def step(self, action: int):
        assert self._last_metrics is not None
        assert self._base_env is not None

        action_spec = self._action_specs[int(action)]
        jump_action_active = action_spec.uses_jump
        jump_started = jump_action_active and not self._prev_jump_action_active
        action_name = action_spec.name
        action_name_tokens = set(action_tokens(action_name))
        frame: np.ndarray | None = None
        for action_mask in action_spec.sequence:
            frame, _base_reward, _base_terminated, _base_truncated, _base_info = self._base_env.step(action_mask)

        assert frame is not None
        self._episode_steps += 1
        current_metrics = self._metrics()
        zone_changed = (current_metrics.world, current_metrics.level, current_metrics.area_offset) != (
            self._last_metrics.world,
            self._last_metrics.level,
            self._last_metrics.area_offset,
        )
        if zone_changed or is_transition_state(current_metrics):
            self._progress_window.clear()
        self._progress_window.append(current_metrics.level_progress)
        self._jump_window.append(1 if jump_started else 0)
        self._speed_window.append(current_metrics.x_speed)
        self._movement_trace.append(self._trace_sample(current_metrics))
        stagnation_window_gain_value = progress_window_gain(self._progress_window)
        stagnation_window_ready = len(self._progress_window) == self._progress_window.maxlen
        jump_window_count = sum(self._jump_window)
        avg_x_speed = sum(self._speed_window) / max(1, len(self._speed_window))
        nearest_enemy = nearest_enemy_ahead(current_metrics)
        nearest_enemy_dx = int(nearest_enemy["dx"]) if nearest_enemy is not None else None
        pipe_stall_detected = is_pipe_stall(current_metrics, stagnation_window_gain_value, self.reward_config)
        progress_delta = current_metrics.progress - self._last_metrics.progress
        if jump_action_active and progress_delta <= 0:
            next_jump_no_progress_steps = self._jump_no_progress_steps + 1
        else:
            next_jump_no_progress_steps = 0
        grounded_run_steps = self._ground_run_steps
        is_moving_right = "right" in action_name_tokens
        is_running_right = is_moving_right and "b" in action_name_tokens
        completed_jump_height: int | None = None
        completed_jump_progress: int | None = None
        completed_jump_landing_delta_y: int | None = None
        current_jump_height = 0
        current_jump_progress = 0
        if jump_started and not self._last_metrics.airborne:
            self._jump_start_y = self._last_metrics.player_y_screen
            self._jump_peak_y = min(self._last_metrics.player_y_screen, current_metrics.player_y_screen)
            self._jump_start_progress = self._last_metrics.progress
        elif self._jump_start_y is not None and self._jump_peak_y is not None:
            self._jump_peak_y = min(self._jump_peak_y, current_metrics.player_y_screen)

        if self._jump_start_y is not None and self._jump_peak_y is not None and self._jump_start_progress is not None:
            current_jump_height = max(0, self._jump_start_y - self._jump_peak_y)
            current_jump_progress = max(0, current_metrics.progress - self._jump_start_progress)
            if not current_metrics.airborne and (self._last_metrics.airborne or jump_started):
                completed_jump_height = current_jump_height
                completed_jump_progress = current_jump_progress
                completed_jump_landing_delta_y = abs(current_metrics.player_y_screen - self._jump_start_y)
                self._last_completed_jump_height = completed_jump_height
                self._last_completed_jump_progress = completed_jump_progress
                self._last_completed_jump_landing_delta_y = completed_jump_landing_delta_y
                self._jump_start_y = None
                self._jump_peak_y = None
                self._jump_start_progress = None
        transition = compute_reward_transition(
            self._last_metrics,
            current_metrics,
            self.reward_config,
            self._stagnation_steps,
            stagnation_window_ready=stagnation_window_ready,
            stagnation_window_gain_value=stagnation_window_gain_value,
            jump_window_count=jump_window_count,
            jump_action_active=jump_action_active,
            jump_started=jump_started,
            was_airborne=self._last_metrics.airborne,
            grounded_run_steps=grounded_run_steps,
            avg_x_speed=avg_x_speed,
            jump_no_progress_steps=next_jump_no_progress_steps,
            is_moving_right=is_moving_right,
            is_running_right=is_running_right,
            completed_jump_height=completed_jump_height,
            completed_jump_progress=completed_jump_progress,
            completed_jump_landing_delta_y=completed_jump_landing_delta_y,
            nearest_enemy_dx=nearest_enemy_dx,
            pipe_stall_detected=pipe_stall_detected,
        )
        self._stagnation_steps = transition.stagnation_steps
        self._furthest_campaign_progress = max(self._furthest_campaign_progress, current_metrics.campaign_progress)
        self._furthest_level_progress = max(self._furthest_level_progress, current_metrics.level_progress)
        grounded_run = is_running_right and not jump_action_active and not current_metrics.airborne
        if grounded_run:
            self._ground_run_steps = min(self._ground_run_steps + 1, 60)
        elif not current_metrics.airborne:
            self._ground_run_steps = 0
        if jump_started and not self._last_metrics.airborne:
            self._ground_run_steps = 0
        self._jump_no_progress_steps = next_jump_no_progress_steps
        self._prev_jump_action_active = jump_action_active
        terminated, truncated, termination_reason = derive_termination_reason(
            current_metrics,
            self.reward_config,
            level_complete=transition.level_complete,
            terminate_on_level_complete=self.terminate_on_level_complete,
            episode_steps=self._episode_steps,
            max_episode_steps=self.max_episode_steps,
            stagnation_steps=self._stagnation_steps,
        )
        self._maybe_record_behavior_events(
            current_metrics,
            nearest_enemy=nearest_enemy,
            pipe_stall_detected=pipe_stall_detected,
            stagnation_steps=transition.stagnation_steps,
            stagnation_window_gain_value=transition.stagnation_window_gain,
            furthest_level_progress=self._furthest_level_progress,
            termination_reason=termination_reason,
        )

        self._last_metrics = current_metrics
        self._last_frame = np.array(frame, copy=True)
        self._render_human_if_needed()
        return (
            self._preprocess(self._last_frame),
            float(transition.reward),
            terminated,
            truncated,
            self._build_info(
                current_metrics,
                action_name=action_spec.name,
                reward_parts=transition.reward_parts,
                termination_reason=termination_reason,
                jump_window_count=transition.jump_window_count,
                stagnation_window_gain_value=transition.stagnation_window_gain,
                avg_x_speed=avg_x_speed,
                current_jump_height=current_jump_height,
                current_jump_progress=current_jump_progress,
                nearest_enemy=nearest_enemy,
                pipe_stall_detected=pipe_stall_detected,
            ),
        )

    def render(self):
        if self._last_frame is None:
            raise RuntimeError("Call reset() before render().")
        if self.render_mode == "human":
            self._render_human_if_needed()
            return None
        return np.array(self._last_frame, copy=True)

    def close(self):
        if self._base_env is not None:
            self._base_env.close()
            self._base_env = None
        self._last_frame = None
        self._last_metrics = None
