import os
import tempfile
import time
from collections import Counter
from copy import deepcopy
from functools import wraps
from threading import Thread

import optuna
from sb3_contrib import TQC
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.vec_env import VecEnv


class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
        best_model_save_path: str | None = None,
        log_path: str | None = None,
    ) -> None:
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elasped time ?
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # make mypy happy
        assert self.model is not None

        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)  # type: ignore[union-attr]
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True


class ParallelTrainCallback(BaseCallback):
    """
    Callback to explore (collect experience) and train (do gradient steps)
    at the same time using two separate threads.
    Normally used with off-policy algorithms and `train_freq=(1, "episode")`.

    TODO:
    - blocking mode: wait for the model to finish updating the policy before collecting new experience
    at the end of a rollout
    - force sync mode: stop training to update to the latest policy for collecting
    new experience

    :param gradient_steps: Number of gradient steps to do before
      sending the new policy
    :param verbose: Verbosity level
    :param sleep_time: Limit the fps in the thread collecting experience.
    """

    def __init__(self, gradient_steps: int = 100, verbose: int = 0, sleep_time: float = 0.0):
        super().__init__(verbose)
        self.batch_size = 0
        self._model_ready = True
        self._model: SAC | TQC
        self.gradient_steps = gradient_steps
        self.process: Thread
        self.model_class: type[SAC] | type[TQC]
        self.sleep_time = sleep_time

    def _init_callback(self) -> None:
        temp_file = tempfile.TemporaryFile()

        # Windows TemporaryFile is not a io Buffer
        # we save the model in the logs/ folder
        if os.name == "nt":
            temp_file = os.path.join("logs", "model_tmp.zip")  # type: ignore[arg-type,assignment]

        # make mypy happy
        assert isinstance(self.model, (SAC, TQC)), f"{self.model} is not supported for parallel training"

        self.model.save(temp_file)  # type: ignore[arg-type]

        # TODO: add support for other algorithms
        for model_class in [SAC, TQC]:
            if isinstance(self.model, model_class):
                self.model_class = model_class  # type: ignore[assignment]
                break

        assert self.model_class is not None, f"{self.model} is not supported for parallel training"
        self._model = self.model_class.load(temp_file)  # type: ignore[arg-type]

        self.batch_size = self._model.batch_size

        # Disable train method
        def patch_train(function):
            @wraps(function)
            def wrapper(*args, **kwargs):
                return

            return wrapper

        # Add logger for parallel training
        self._model.set_logger(self.model.logger)
        self.model.train = patch_train(self.model.train)  # type: ignore[assignment]

        # Hack: Re-add correct values at save time
        def patch_save(function):
            @wraps(function)
            def wrapper(*args, **kwargs):
                return self._model.save(*args, **kwargs)

            return wrapper

        self.model.save = patch_save(self.model.save)  # type: ignore[assignment]

    def train(self) -> None:
        self._model_ready = False

        self.process = Thread(target=self._train_thread, daemon=True)
        self.process.start()

    def _train_thread(self) -> None:
        self._model.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
        self._model_ready = True

    def _on_step(self) -> bool:
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)
        return True

    def _on_rollout_end(self) -> None:
        # Make mypy happy
        assert isinstance(self.model, (SAC, TQC))

        if self._model_ready:
            self._model.replay_buffer = deepcopy(self.model.replay_buffer)
            self.model.set_parameters(deepcopy(self._model.get_parameters()))  # type: ignore[arg-type]
            self.model.actor = self.model.policy.actor  # type: ignore[union-attr, attr-defined, assignment]
            if self.num_timesteps >= self._model.learning_starts:
                self.train()
            # Do not wait for the training loop to finish
            # self.process.join()

    def _on_training_end(self) -> None:
        # Wait for the thread to terminate
        if self.process is not None:
            if self.verbose > 0:
                print("Waiting for training thread to terminate")
            self.process.join()


class RawStatisticsCallback(BaseCallback):
    """
    Callback used for logging raw episode data (return and episode length).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Custom counter to reports stats
        # (and avoid reporting multiple values for the same step)
        self._timesteps_counter = 0
        self._tensorboard_writer = None

    def _init_callback(self) -> None:
        assert self.logger is not None
        # Retrieve tensorboard writer to not flood the logger output
        for out_format in self.logger.output_formats:
            if isinstance(out_format, TensorBoardOutputFormat):
                self._tensorboard_writer = out_format
        assert self._tensorboard_writer is not None, "You must activate tensorboard logging when using RawStatisticsCallback"

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                logger_dict = {
                    "raw/rollouts/episodic_return": info["episode"]["r"],
                    "raw/rollouts/episodic_length": info["episode"]["l"],
                }
                exclude_dict = {key: None for key in logger_dict.keys()}
                self._timesteps_counter += info["episode"]["l"]
                self._tensorboard_writer.write(logger_dict, exclude_dict, self._timesteps_counter)

        return True


class NESSMBTrainingStatsCallback(BaseCallback):
    """
    Callback for compact SMB progress logging in the zoo trainer.
    """

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self._rollout_action_counts: Counter[str] = Counter()
        self._rollout_button_counts: Counter[str] = Counter()
        self._rollout_reward_parts: Counter[str] = Counter()
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._max_world = 0
        self._max_level = 0
        self._max_level_x = 0
        self._max_campaign_progress = 0
        self._latest_world = 0
        self._latest_level = 0
        self._latest_level_x = 0
        self._latest_furthest_x = 0
        self._latest_ground_time_ratio = 0.0
        self._last_termination_reason: str | None = None
        self._last_event_id = 0

    @staticmethod
    def _button_tokens(action_name: str) -> tuple[str, ...]:
        if not action_name or action_name == "noop":
            return ()
        allowed_tokens = {"right", "left", "down", "up", "a", "b"}
        return tuple(
            token
            for token in (part.lower() for part in action_name.split("_"))
            if token in allowed_tokens
        )

    @staticmethod
    def _event_location(last_event: dict[str, object]) -> str:
        return f"W{int(last_event.get('world_display', 1) or 1)}-{int(last_event.get('level_display', 1) or 1)}"

    @staticmethod
    def _event_detail(last_event: dict[str, object]) -> str:
        detail = ""
        if "enemy_label" in last_event:
            detail += f" enemy={last_event['enemy_label']}"
        if "enemy_dx" in last_event:
            detail += f" dx={last_event['enemy_dx']}"
        if "x_speed" in last_event:
            detail += f" x_speed={last_event['x_speed']}"
        if "stagnation_steps" in last_event:
            detail += f" stagnation_steps={last_event['stagnation_steps']}"
        if "stagnation_window_gain" in last_event:
            detail += f" gain={last_event['stagnation_window_gain']}"
        if "furthest_window_gain" in last_event:
            detail += f" furthest_gain={last_event['furthest_window_gain']}"
        if "direction_changes" in last_event:
            detail += f" dir_changes={last_event['direction_changes']}"
        if "furthest_gap" in last_event:
            detail += f" furthest_gap={last_event['furthest_gap']}"
        if "hop_count" in last_event:
            detail += f" hop_count={last_event['hop_count']}"
        if "timer_window" in last_event:
            detail += f" timer_window={last_event['timer_window']}"
        if "speed_trace" in last_event:
            detail += f" speed_trace={last_event['speed_trace']}"
        if "progress_trace" in last_event:
            detail += f" progress_trace={last_event['progress_trace']}"
        return detail

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            action_name = info.get("action_name")
            if action_name:
                self._rollout_action_counts[action_name] += 1
                for token in self._button_tokens(action_name):
                    self._rollout_button_counts[token] += 1

            reward_parts = info.get("reward_parts")
            if isinstance(reward_parts, dict):
                for name, value in reward_parts.items():
                    self._rollout_reward_parts[str(name)] += float(value)

            world_display = int(info.get("world_display", 0) or 0)
            level_display = int(info.get("level_display", 0) or 0)
            level_progress = int(info.get("level_progress", 0) or 0)
            furthest_level_progress = int(info.get("furthest_level_progress", 0) or 0)
            campaign_progress = int(info.get("furthest_progress", 0) or 0)
            ground_time_ratio = float(info.get("ground_time_ratio", 0.0) or 0.0)

            self._latest_world = world_display or self._latest_world
            self._latest_level = level_display or self._latest_level
            self._latest_level_x = level_progress
            self._latest_furthest_x = furthest_level_progress
            self._latest_ground_time_ratio = ground_time_ratio

            self._max_world = max(self._max_world, world_display)
            self._max_level = max(self._max_level, level_display)
            self._max_level_x = max(self._max_level_x, level_progress, furthest_level_progress)
            self._max_campaign_progress = max(self._max_campaign_progress, campaign_progress)

            if "episode" in info:
                self._episode_rewards.append(float(info["episode"]["r"]))
                self._episode_lengths.append(int(info["episode"]["l"]))

            termination_reason = info.get("termination_reason")
            if termination_reason:
                self._last_termination_reason = str(termination_reason)

            last_event = info.get("last_event")
            if isinstance(last_event, dict):
                event_id = int(last_event.get("id", 0) or 0)
                if event_id > self._last_event_id:
                    self._last_event_id = event_id
                    if self.verbose > 0:
                        location = self._event_location(last_event)
                        detail = self._event_detail(last_event)
                        print(
                            f"[NES-SMB event] {location} type={last_event.get('type')} "
                            f"step={last_event.get('episode_steps')} x={last_event.get('level_progress')}{detail}"
                        )

            if info.get("debug_reward_line"):
                nearest_enemy = info.get("nearest_enemy")
                enemy = ""
                if isinstance(nearest_enemy, dict):
                    enemy = f" enemy={nearest_enemy.get('label')} dx={nearest_enemy.get('dx')}"
                parts = info.get("debug_reward_parts") or {}
                reward_summary = " ".join(
                    f"{name}:{float(value):+.2f}" for name, value in sorted(parts.items(), key=lambda item: item[0])
                )
                print(
                    f"[NES-SMB reward] W{world_display or 1}-{level_display or 1} "
                    f"step={info.get('episode_steps')} x={level_progress} action={action_name} "
                    f"raw_dx={info.get('raw_progress_delta')} new_dx={info.get('new_furthest_delta')} "
                    f"stagnation={info.get('stagnation_steps')} jump_window={info.get('jump_window_count')}"
                    f"{enemy} parts[{reward_summary or 'none'}]"
                )

        return True

    def _on_rollout_end(self) -> None:
        if self._rollout_action_counts:
            for action_name, count in sorted(self._rollout_action_counts.items()):
                self.logger.record(f"actions/{action_name}", count)

        if self._rollout_button_counts:
            for button_name, count in sorted(self._rollout_button_counts.items()):
                self.logger.record(f"buttons/{button_name}", count)

        if self._rollout_reward_parts:
            for part_name, total in sorted(self._rollout_reward_parts.items()):
                self.logger.record(f"reward_parts/{part_name}", total)

        self.logger.record("progress/max_world", self._max_world)
        self.logger.record("progress/max_level", self._max_level)
        self.logger.record("progress/max_level_x", self._max_level_x)
        self.logger.record("progress/max_campaign_progress", self._max_campaign_progress)
        self.logger.record("ground/ground_time_ratio", self._latest_ground_time_ratio)

        if self._episode_rewards:
            self.logger.record("rollout/ep_rew_mean", sum(self._episode_rewards) / len(self._episode_rewards))
            self.logger.record("rollout/ep_len_mean", sum(self._episode_lengths) / len(self._episode_lengths))

        if self.verbose > 0:
            action_summary = " ".join(
                f"{name}:{count}"
                for name, count in sorted(self._rollout_action_counts.items(), key=lambda item: (-item[1], item[0]))
            )
            button_summary = " ".join(
                f"{name}:{count}"
                for name, count in sorted(self._rollout_button_counts.items(), key=lambda item: (-item[1], item[0]))
            )
            reward_summary = " ".join(
                f"{name}:{total:+.1f}"
                for name, total in sorted(
                    self._rollout_reward_parts.items(),
                    key=lambda item: (-abs(item[1]), item[0]),
                )[:8]
            )
            location = f"W{self._latest_world or 1}-{self._latest_level or 1}"
            reason = f" reason={self._last_termination_reason}" if self._last_termination_reason else ""
            print(
                f"[NES-SMB] {location} x={self._latest_level_x} furthest_x={self._latest_furthest_x} "
                f"best_campaign={self._max_campaign_progress}{reason} "
                f"ground_time={self._latest_ground_time_ratio:.2f} "
                f"buttons[{button_summary or 'none'}] actions[{action_summary or 'none'}] "
                f"rewards[{reward_summary or 'none'}]"
            )

        self._rollout_action_counts.clear()
        self._rollout_button_counts.clear()
        self._rollout_reward_parts.clear()
        self._episode_rewards.clear()
        self._episode_lengths.clear()
        self._last_termination_reason = None
