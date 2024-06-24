#!/bin/bash

base_model="code-t5-small"
sample="optimal_dist_12.0"
sample_name="optimal_dist_12"
tmux new-session -t "sokoban_lm_${sample_name}_$base_model" -d -s
tmux send-keys -t "sokoban_lm_${sample_name}_$base_model" "conda activate reasoners" C-m
tmux send-keys -t "sokoban_lm_${sample_name}_$base_model" "CUDA_VISIBLE_DEVICES=3 ./launch_sokoban.sh lm $sample $base_model" C-m

tmux new-session -t "sokoban_l2_${sample_name}_$base_model" -d -s
tmux send-keys -t "sokoban_l2_${sample_name}_$base_model" "conda activate reasoners" C-m
tmux send-keys -t "sokoban_l2_${sample_name}_$base_model" "CUDA_VISIBLE_DEVICES=1 ./launch_sokoban.sh l2 $sample $base_model" C-m

tmux new-session -t "maze_lm_${sample_name}_$base_model" -d -s
tmux send-keys -t "maze_lm_${sample_name}_$base_model" "conda activate reasoners" C-m
tmux send-keys -t "maze_lm_${sample_name}_$base_model" "CUDA_VISIBLE_DEVICES=1 ./launch_maze.sh lm $sample $base_model" C-m

tmux new-session -t "maze_l2_${sample_name}_$base_model" -d -s
tmux send-keys -t "maze_l2_${sample_name}_$base_model" "conda activate reasoners" C-m
tmux send-keys -t "maze_l2_${sample_name}_$base_model" "CUDA_VISIBLE_DEVICES=2 ./launch_maze.sh l2 $sample $base_model" C-m
