# Autoresearch: Optimizing NES-SMB-RL training progress

## Objective
Tune hyperparameters in hyperparams/python/ppo_nes_smb.py to improve training completion time and final performance on NES-SMB benchmark.

## Metrics
- **Primary**: max_level (unitless, higher is better)
- **Secondary**: training_time_seconds, final_reward

## How to Run
`mise run train-nes-watch`

## Files in Scope
- hyperparams/python/ppo_nes_smb.py
- autoresearch.sh
- autoresearch.config.json

## Constraints
- Must use nerf-based algorithm (ppo is fixed)
- No new dependencies
- Must complete training within 300s

## What's Been Tried
- Initial config timed out after 600s
- Training needs speed optimization
- Potential areas: batch_size, n_timesteps, frame_stack, learning_rate

## Ideas for Next Steps
- Reduce n_timesteps from 200k to 50k
- Decrease frame_stack from 8 to 4
- Lower learning_rate from 2.5e-4 to 1e-4
- Reduce n_epochs from 4 to 2