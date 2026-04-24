from __future__ import annotations

import os

# Edit this file when you want to flip SMB training incentives on/off.
# Use 1 for enabled and 0 for disabled.

ROM_PATH = os.environ.get(
    "NES_SMB_ROM_PATH",
    "/Users/justinedwards/git/NES-SMB-RL/nes-py/nes_py/tests/games/super-mario-bros-1.nes",
)

# Runtime visibility/debugging.
VISIBLE_TRAINING = 1
DEBUG_REWARDS = 1

# Action-space switches.
ENABLE_LONG_JUMPS = 1
ENABLE_LEFT_JUMP_ACTIONS = 0

# Reward/incentive switches.
ENABLE_PROGRESS_REWARD = 1
ENABLE_GROUNDED_NEW_PROGRESS = 1
ENABLE_TIME_PENALTY = 1
ENABLE_DEATH_PENALTY = 1
ENABLE_COIN_REWARD = 1
ENABLE_POWERUP_REWARD = 1
ENABLE_RUN_GROUND_BONUS = 1
ENABLE_JUMP_SHAPING = 1
ENABLE_GOOD_JUMP_BONUS = 1
ENABLE_STOMP_BONUS = 1
ENABLE_ENEMY_PENALTY = 1
ENABLE_PIPE_PENALTY = 1
ENABLE_JUMP_WINDOW_PENALTY = 1
ENABLE_HOP_CLUSTER_PENALTY = 1
ENABLE_OSCILLATION_PENALTY = 1
ENABLE_STAGNATION_PENALTY = 1
ENABLE_HOP_SPAM_DEATH = 0

# Episode and stagnation settings.
MAX_EPISODE_STEPS = 12000
STAGNATION_STEPS = 240
STAGNATION_WINDOW = 48
STAGNATION_MIN_PROGRESS = 24
STAGNATION_FURTHEST_MIN_PROGRESS = 12
STAGNATION_PENALTY = 1.0
OSCILLATION_PENALTY = 0.2

# Jump-shaping settings.
JUMP_PENALTY_WINDOW = 32
JUMP_PENALTY_THRESHOLD = 8
JUMP_PENALTY_SCALE = 0.03
JUMP_NO_PROGRESS_PENALTY = 0.08
JUMP_IDLE_PENALTY = 0.12
JUMP_START_PENALTY = 0.3
SHORT_JUMP_PENALTY = 0.06
SHORT_JUMP_HEIGHT_THRESHOLD = 10
GOOD_JUMP_BONUS = 0.12
GOOD_JUMP_HEIGHT_THRESHOLD = 14
GOOD_JUMP_PROGRESS_THRESHOLD = 12
LANDING_SPOT_BONUS = 0.08
LANDING_Y_CHANGE_THRESHOLD = 6
LANDING_PROGRESS_THRESHOLD = 12
HOP_CLUSTER_PENALTY_SCALE = 0.16
HOP_CLUSTER_COUNT_THRESHOLD = 4
HOP_CLUSTER_PROGRESS_THRESHOLD = 24

# Ground/run incentives.
AIRBORNE_PROGRESS_SCALE = 0.5
GROUNDED_NEW_PROGRESS_SCALE = 1.0 / 30.0
RUN_GROUND_BONUS = 0.04
FULL_SPEED_GROUND_BONUS = 0.12
FULL_SPEED_X_THRESHOLD = 47
FULL_SPEED_RUN_STEPS = 12

# Enemy/pipe behavior.
STOMP_BONUS = 3.0
ENEMY_IGNORE_PENALTY = 0.16
ENEMY_IMMEDIATE_DX_THRESHOLD = 18
PIPE_STALL_CHOICE_PENALTY = 0.2
PIPE_STALL_SCREEN_X_THRESHOLD = 100
PIPE_STALL_SPEED_THRESHOLD = 5
PIPE_STALL_PROGRESS_THRESHOLD = 12

# Macro action timing.
FRAMES_PER_ACTION = 2
MACRO_JUMP_HOLD_FRAMES = 6
MACRO_JUMP_RELEASE_FRAMES = 1

# Terminal reward debug logging.
DEBUG_REWARD_INTERVAL = 16
DEBUG_REWARD_MIN_ABS = 0.05
DEBUG_ENEMY_DISTANCE = 64


def _enabled(value: int) -> bool:
    return bool(value)


BASE_ENV_KWARGS = dict(
    rom_path=ROM_PATH,
    frames_per_action=FRAMES_PER_ACTION,
    max_episode_steps=MAX_EPISODE_STEPS,
    enable_progress_reward=_enabled(ENABLE_PROGRESS_REWARD),
    enable_grounded_new_progress=_enabled(ENABLE_GROUNDED_NEW_PROGRESS),
    enable_time_penalty=_enabled(ENABLE_TIME_PENALTY),
    enable_death_penalty=_enabled(ENABLE_DEATH_PENALTY),
    enable_coin_reward=_enabled(ENABLE_COIN_REWARD),
    enable_powerup_reward=_enabled(ENABLE_POWERUP_REWARD),
    enable_run_ground_bonus=_enabled(ENABLE_RUN_GROUND_BONUS),
    enable_jump_shaping=_enabled(ENABLE_JUMP_SHAPING),
    enable_good_jump_bonus=_enabled(ENABLE_GOOD_JUMP_BONUS),
    enable_stomp_bonus=_enabled(ENABLE_STOMP_BONUS),
    enable_enemy_penalty=_enabled(ENABLE_ENEMY_PENALTY),
    enable_pipe_penalty=_enabled(ENABLE_PIPE_PENALTY),
    enable_jump_window_penalty=_enabled(ENABLE_JUMP_WINDOW_PENALTY),
    enable_hop_cluster_penalty=_enabled(ENABLE_HOP_CLUSTER_PENALTY),
    enable_oscillation_penalty=_enabled(ENABLE_OSCILLATION_PENALTY),
    enable_stagnation_penalty=_enabled(ENABLE_STAGNATION_PENALTY),
    enable_hop_spam_death=_enabled(ENABLE_HOP_SPAM_DEATH),
    airborne_progress_scale=AIRBORNE_PROGRESS_SCALE,
    stagnation_steps=STAGNATION_STEPS,
    stagnation_window=STAGNATION_WINDOW,
    stagnation_min_progress=STAGNATION_MIN_PROGRESS,
    stagnation_furthest_min_progress=STAGNATION_FURTHEST_MIN_PROGRESS,
    stagnation_penalty=STAGNATION_PENALTY,
    oscillation_penalty=OSCILLATION_PENALTY,
    jump_penalty_window=JUMP_PENALTY_WINDOW,
    jump_penalty_threshold=JUMP_PENALTY_THRESHOLD,
    jump_penalty_scale=JUMP_PENALTY_SCALE,
    jump_no_progress_penalty=JUMP_NO_PROGRESS_PENALTY,
    jump_idle_penalty=JUMP_IDLE_PENALTY,
    jump_start_penalty=JUMP_START_PENALTY,
    short_jump_penalty=SHORT_JUMP_PENALTY,
    short_jump_height_threshold=SHORT_JUMP_HEIGHT_THRESHOLD,
    good_jump_bonus=GOOD_JUMP_BONUS,
    good_jump_height_threshold=GOOD_JUMP_HEIGHT_THRESHOLD,
    good_jump_progress_threshold=GOOD_JUMP_PROGRESS_THRESHOLD,
    landing_spot_bonus=LANDING_SPOT_BONUS,
    landing_y_change_threshold=LANDING_Y_CHANGE_THRESHOLD,
    landing_progress_threshold=LANDING_PROGRESS_THRESHOLD,
    hop_cluster_penalty_scale=HOP_CLUSTER_PENALTY_SCALE,
    hop_cluster_count_threshold=HOP_CLUSTER_COUNT_THRESHOLD,
    hop_cluster_progress_threshold=HOP_CLUSTER_PROGRESS_THRESHOLD,
    grounded_new_progress_scale=GROUNDED_NEW_PROGRESS_SCALE,
    run_ground_bonus=RUN_GROUND_BONUS,
    full_speed_ground_bonus=FULL_SPEED_GROUND_BONUS,
    full_speed_x_threshold=FULL_SPEED_X_THRESHOLD,
    full_speed_run_steps=FULL_SPEED_RUN_STEPS,
    stomp_bonus=STOMP_BONUS,
    enemy_ignore_penalty=ENEMY_IGNORE_PENALTY,
    enemy_immediate_dx_threshold=ENEMY_IMMEDIATE_DX_THRESHOLD,
    pipe_stall_choice_penalty=PIPE_STALL_CHOICE_PENALTY,
    pipe_stall_screen_x_threshold=PIPE_STALL_SCREEN_X_THRESHOLD,
    pipe_stall_speed_threshold=PIPE_STALL_SPEED_THRESHOLD,
    pipe_stall_progress_threshold=PIPE_STALL_PROGRESS_THRESHOLD,
    enable_long_jumps=_enabled(ENABLE_LONG_JUMPS),
    enable_left_jump_actions=_enabled(ENABLE_LEFT_JUMP_ACTIONS),
    macro_jump_hold_frames=MACRO_JUMP_HOLD_FRAMES,
    macro_jump_release_frames=MACRO_JUMP_RELEASE_FRAMES,
    debug_rewards=_enabled(DEBUG_REWARDS),
    debug_reward_interval=DEBUG_REWARD_INTERVAL,
    debug_reward_min_abs=DEBUG_REWARD_MIN_ABS,
    debug_enemy_distance=DEBUG_ENEMY_DISTANCE,
)

TRAIN_ENV_KWARGS = {
    **BASE_ENV_KWARGS,
    "random_noop_max": 30,
    "render_mode": "human" if _enabled(VISIBLE_TRAINING) else "rgb_array",
}

EVAL_ENV_KWARGS = {
    **BASE_ENV_KWARGS,
    "random_noop_max": 0,
    "render_mode": "rgb_array",
    "debug_rewards": False,
}
