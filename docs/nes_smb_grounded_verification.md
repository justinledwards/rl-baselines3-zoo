# NES-SMB Grounded Verification

This env has two kinds of tests:

- Synthetic RAM tests in `tests/test_nes_smb_env.py` prove pure reward math and RAM parsing in isolation.
- ROM-grounded tests in `tests/test_nes_smb_env_rom_states.py` step a real `nes-py` ROM and prove that action sequences cause the RAM states the reward code assumes.

The ROM tests are named as claims. If a reward depends on a game-state assumption, prefer adding a `test_rom_*` case with the action sequence that makes the assumption true.

## Grounded Checks

| Assumption | ROM-grounded test |
| --- | --- |
| Reset starts in the level on solid ground; no-op does not fake airborne/progress | `test_rom_reset_starts_grounded_and_noop_stays_grounded` |
| Pressing jump changes RAM to airborne, raises Y position, and landing returns to grounded | `test_rom_jump_action_sets_airborne_then_lands_grounded` |
| Bad standstill jumps produce jump-start/no-progress/idle penalties | `test_rom_jump_action_sets_airborne_then_lands_grounded` |
| Grounded running ramps to full speed and pays run/full-speed bonuses | `test_rom_grounded_running_reaches_full_speed_and_receives_ground_bonus` |
| Ground/run bonuses do not pay when pressing into an obstruction without new furthest progress | `test_rom_ground_run_bonus_stops_when_pressed_against_pipe_without_new_furthest_progress` |
| Enemy detection uses level-space position, not screen-space position | `test_rom_enemy_is_detected_by_level_position_and_critical_enemy_penalty_fires` |
| Critical enemy proximity gives the ignore penalty while grounded | `test_rom_enemy_is_detected_by_level_position_and_critical_enemy_penalty_fires` |
| Enemy collision terminates as death and logs death-near-enemy | `test_rom_enemy_collision_terminates_with_death_near_enemy_event` |
| Enemy spawning does not falsely trigger powerup reward | `test_rom_enemy_spawn_does_not_trigger_powerup_reward` |
| Coin reward only appears when the coin counter increments in the real game | `test_rom_coin_reward_appears_when_coin_counter_increments` |
| Pipe stalls log an event and pay pipe-stall penalty after a real obstruction | `test_rom_pipe_stall_and_stagnation_events_are_logged_after_real_stall` |
| Stagnation logging starts after sustained real no-progress behavior, not on the first acceleration frame | `test_rom_pipe_stall_and_stagnation_events_are_logged_after_real_stall` |
| Hop-spam death can terminate when explicitly enabled | `test_rom_hop_spam_death_can_terminate_when_enabled` |
| Landing on a higher surface is still grounded, not airborne | `test_rom_landing_on_higher_surface_is_grounded_not_airborne` |

## Synthetic-Only Checks

These are still synthetic because they need precise isolated RAM transitions or long playthrough states:

- Level-complete/flag bonus.
- Actual powerup pickup bonus.
- Game-complete termination.
- Stomp bonus from enemy-state transition.
- Feature-flag disabling behavior.
- Action-space filtering.

The linked TAS repository includes `.fm2` traces and documents frame-level validation against original ROM state. Those traces are the right next source for long-horizon grounded tests such as powerup pickup, level completion, and game completion.
