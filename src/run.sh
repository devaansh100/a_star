#!/bin/bash

base_model="code-t5"
sample="semdedup"
sample_name="semdedup"
tmux new-session -t "stp_lm_${sample_name}" -d -s
tmux send-keys -t "stp_lm_${sample_name}" "conda activate reasoners" C-m
tmux send-keys -t "stp_lm_${sample_name}" "CUDA_VISIBLE_DEVICES=1 ./launch_stp.sh lm $sample $base_model" C-m

# sample="optimal_dist_lin_2.0"
# sample_name="optimal_dist_lin_2"
# tmux new-session -t "maze_l2_${sample_name}" -d -s
# tmux send-keys -t "maze_l2_${sample_name}" "conda activate reasoners" C-m
# tmux send-keys -t "maze_l2_${sample_name}" "CUDA_VISIBLE_DEVICES=1 ./launch_maze.sh l2 $sample $base_model" C-m

# sample="optimal_dist_lin_0.8"
# sample_name="optimal_dist_lin_0_8"
# tmux new-session -t "sokoban_l2_${sample_name}" -d -s
# tmux send-keys -t "sokoban_l2_${sample_name}" "conda activate reasoners" C-m
# tmux send-keys -t "sokoban_l2_${sample_name}" "CUDA_VISIBLE_DEVICES=2 ./launch_sokoban.sh l2 $sample $base_model" C-m


# sample="optimal_dist_12.0"
# sample_name="optimal_dist_12"
# tmux new-session -t "fin-stp_alpha_l2_${sample_name}" -d -s
# tmux send-keys -t "fin-stp_alpha_l2_${sample_name}" "conda activate reasoners" C-m
# tmux send-keys -t "fin-stp_alpha_l2_${sample_name}" "CUDA_VISIBLE_DEVICES=1 ./launch_stp.sh l2 $sample $base_model" C-m

# sample="optimal_dist_10.0"
# sample_name="optimal_dist_10"
# tmux new-session -t "stp_l2_${sample_name}" -d -s
# tmux send-keys -t "stp_l2_${sample_name}" "conda activate reasoners" C-m
# tmux send-keys -t "stp_l2_${sample_name}" "CUDA_VISIBLE_DEVICES=2 ./launch_stp.sh l2 $sample $base_model" C-m

# sample="optimal_dist_3.0"
# sample_name="optimal_dist_3"
# tmux new-session -t "stp_lm_${sample_name}" -d -s
# tmux send-keys -t "stp_lm_${sample_name}" "conda activate reasoners" C-m
# tmux send-keys -t "stp_lm_${sample_name}" "CUDA_VISIBLE_DEVICES=3 ./launch_stp.sh lm $sample $base_model" C-m
