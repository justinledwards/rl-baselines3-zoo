#!/bin/bash
set -euo pipefail

# Run training and capture output
output=$(mise run train-nes-watch 2>&1)

# Print full output for debugging
echo "$output"

# Try to extract max_level from training logs or final output
max_level=$(echo "$output" | grep -i "max_level\|best_campaign\|level" | tail -1 | grep -oP '[0-9]+' | tail -1 || echo "0")
echo "METRIC max_level=$max_level"

# Also extract rewards for secondary monitoring
reward=$(echo "$output" | grep "ep_rew_mean" | tail -1 | grep -oP '[0-9]+\.?[0-9]*' || echo "0")
echo "METRIC reward=$reward"
