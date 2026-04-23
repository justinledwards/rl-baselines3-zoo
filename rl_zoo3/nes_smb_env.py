from __future__ import annotations

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


@dataclass(frozen=True)
class SMBRamAddresses:
    player_page: int = 0x006D
    player_x: int = 0x0086
    player_x_screen: int = 0x0755
    player_state: int = 0x000E
    player_mode: int = 0x000F
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
    flag_bonus: float = 500.0
    coin_bonus: float = 1.0
    stagnation_penalty: float = 1.0
    stagnation_steps: int = 240
    death_states: tuple[int, ...] = (6, 11)
    time_progress_weight: float = 0.001


@dataclass(frozen=True)
class SMBRamMetrics:
    player_state: int
    player_mode: int
    player_page: int
    player_x: int
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
            "player_page": self.player_page,
            "player_x": self.player_x,
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


def build_metrics_from_ram(ram: np.ndarray, addresses: SMBRamAddresses | None = None) -> SMBRamMetrics:
    addr = addresses or SMBRamAddresses()
    player_page = int(ram[addr.player_page])
    player_x = int(ram[addr.player_x])
    player_x_screen = int(ram[addr.player_x_screen])
    world = int(ram[addr.world])
    level = int(ram[addr.level_dash])
    progress = player_page * SCREEN_WIDTH + player_x
    level_progress = player_page * SCREEN_WIDTH + player_x_screen
    timer = int(ram[addr.timer_digit_1]) * 100 + int(ram[addr.timer_digit_2]) * 10 + int(ram[addr.timer_digit_3])
    campaign_progress = ((world * 4) + level) * 4096 + level_progress
    return SMBRamMetrics(
        player_state=int(ram[addr.player_state]),
        player_mode=int(ram[addr.player_mode]),
        player_page=player_page,
        player_x=player_x,
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
    )


def is_dead(metrics: SMBRamMetrics, reward_config: RewardConfig | None = None) -> bool:
    config = reward_config or RewardConfig()
    return metrics.player_state in config.death_states


def is_transition_state(metrics: SMBRamMetrics) -> bool:
    return metrics.game_mode != PLAYING_GAME_MODE or metrics.player_state in TRANSITION_PLAYER_STATES


def is_game_complete(metrics: SMBRamMetrics) -> bool:
    return metrics.game_mode == END_GAME_MODE and (metrics.world, metrics.level) == FINAL_STAGE


def compute_reward_transition(
    previous: SMBRamMetrics,
    current: SMBRamMetrics,
    reward_config: RewardConfig,
    stagnation_steps: int,
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

    if is_transition_state(current) or zone_changed or progress_delta != 0:
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

    if level_changed:
        reward += reward_config.flag_bonus
        reward_parts["level_complete"] = reward_config.flag_bonus

    coin_delta = current.coins - previous.coins
    if coin_delta > 0:
        coin_reward = coin_delta * reward_config.coin_bonus
        reward += coin_reward
        reward_parts["coins"] = coin_reward

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
    return tuple(token.lower() for token in action_name.split("_"))


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
        terminate_on_level_complete: bool = False,
        random_noop_max: int = 30,
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
        self.reward_config = RewardConfig(stagnation_steps=stagnation_steps)
        self._addresses = SMBRamAddresses()

        height, width = resize_shape[1], resize_shape[0]
        channels = 1 if grayscale else 3
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, channels), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(SLIM_ACTIONS))

        self._base_env = None
        self._agent_env = None
        self._last_frame: np.ndarray | None = None
        self._last_metrics: SMBRamMetrics | None = None
        self._episode_steps = 0
        self._stagnation_steps = 0
        self._furthest_campaign_progress = 0
        self._furthest_level_progress = 0

        self._load_backend()
        self._boot_to_level_start()

    def _load_backend(self) -> None:
        try:
            from nes_py import NESEnv
            from nes_py.wrappers import JoypadSpace
        except ImportError as exc:
            raise ImportError(
                "NES-SMB-v0 requires nes-py in the active environment. "
                "Run `mise run setup` in rl-baselines3-zoo to install the sibling checkout."
            ) from exc

        self._base_env = NESEnv(str(self.rom_path))
        action_layouts = [list(buttons) for _name, buttons, _mask in SLIM_ACTIONS]
        self._agent_env = JoypadSpace(self._base_env, action_layouts)

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
                "termination_reason": termination_reason,
                "reward_parts": reward_parts,
                "render_mode": self.render_mode,
            }
        )
        return info

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(name for name, _buttons, _mask in SLIM_ACTIONS)

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
        self._last_frame = np.array(frame, copy=True)

        self._render_human_if_needed()
        return self._preprocess(self._last_frame), self._build_info(
            metrics,
            action_name="noop",
            reward_parts={},
            termination_reason=None,
        )

    def step(self, action: int):
        assert self._agent_env is not None
        assert self._last_metrics is not None

        action_name, _buttons, _mask = SLIM_ACTIONS[int(action)]
        frame: np.ndarray | None = None
        for _ in range(self.frames_per_action):
            frame, _base_reward, _base_terminated, _base_truncated, _base_info = self._agent_env.step(int(action))

        assert frame is not None
        self._episode_steps += 1
        current_metrics = self._metrics()
        transition = compute_reward_transition(
            self._last_metrics,
            current_metrics,
            self.reward_config,
            self._stagnation_steps,
        )
        self._stagnation_steps = transition.stagnation_steps
        self._furthest_campaign_progress = max(self._furthest_campaign_progress, current_metrics.campaign_progress)
        self._furthest_level_progress = max(self._furthest_level_progress, current_metrics.level_progress)
        terminated, truncated, termination_reason = derive_termination_reason(
            current_metrics,
            self.reward_config,
            level_complete=transition.level_complete,
            terminate_on_level_complete=self.terminate_on_level_complete,
            episode_steps=self._episode_steps,
            max_episode_steps=self.max_episode_steps,
            stagnation_steps=self._stagnation_steps,
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
                action_name=action_name,
                reward_parts=transition.reward_parts,
                termination_reason=termination_reason,
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
        self._agent_env = None
        self._last_frame = None
        self._last_metrics = None
