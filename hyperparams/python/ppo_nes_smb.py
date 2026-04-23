from __future__ import annotations

from rl_zoo3.callbacks import NESSMBTrainingStatsCallback

hyperparams = {
    "NES-SMB-v0": dict(
        policy="CnnPolicy",
        n_envs=1,
        n_timesteps=200000.0,
        frame_stack=8,
        n_steps=1024,
        batch_size=256,
        n_epochs=4,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        callback=[NESSMBTrainingStatsCallback(verbose=1)],
    )
}
