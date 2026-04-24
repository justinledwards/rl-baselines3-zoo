from __future__ import annotations

from hyperparams.python.nes_smb_env_preset import EVAL_ENV_KWARGS, TRAIN_ENV_KWARGS
from rl_zoo3.callbacks import NESSMBTrainingStatsCallback

hyperparams = {
    "NES-SMB-v0": dict(
        policy="CnnLstmPolicy",
        n_envs=1,
        n_timesteps=500000.0,
        frame_stack=4,
        n_steps=512,
        batch_size=128,
        n_epochs=4,
        learning_rate="lin_2.5e-4",
        gamma=0.99,
        gae_lambda=0.95,
        clip_range="lin_0.1",
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        env_kwargs=TRAIN_ENV_KWARGS,
        eval_env_kwargs=EVAL_ENV_KWARGS,
        callback=[NESSMBTrainingStatsCallback(verbose=1)],
    )
}
