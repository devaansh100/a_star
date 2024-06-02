#!/bin/bash
# python main.py --create-data 6000 1000 1000 10 --domain maze --dataset maze-grade --train-seqs 8 --sample optimal
# python main.py --create-data 0 0 0 10 --domain maze --dataset maze-grade --train-seqs 8 --sample opt_dec --sampled-nodes 4
# python main.py --create-data 0 0 0 10 --domain maze --dataset maze-grade --train-seqs 8 --sample rand_opt --sampled-nodes 4
# python main.py --create-data 0 0 1000 15 --domain maze --dataset maze-eval
# python main.py --create-data 0 0 1000 20 --domain maze --dataset maze-eval
# python main.py --create-data 0 0 1000 25 --domain maze --dataset maze-eval
# mv ../datasets/maze-grade/test/* ../datasets/maze-eval/test
# rmdir ../datasets/maze-grade/test/
# samples=("optimal_hard" "optimal_easy" "optimal_med")
# samples=("optimal_easy_med")

# tmux new-session -t "sokoban_lm_grade_med_hard" -d -s
# tmux send-keys -t "sokoban_lm_grade_med_hard" "conda activate reasoners" C-m
# tmux send-keys -t "sokoban_lm_grade_med_hard" "CUDA_VISIBLE_DEVICES=3 ./launch_sokoban.sh lm" C-m

# tmux new-session -t "sokoban_l2_grade_med_hard" -d -s
# tmux send-keys -t "sokoban_l2_grade_med_hard" "conda activate reasoners" C-m
# tmux send-keys -t "sokoban_l2_grade_med_hard" "CUDA_VISIBLE_DEVICES=3 ./launch_sokoban.sh l2" C-m

# tmux new-session -t "maze_l2_grade_dist" -d -s
# tmux send-keys -t "maze_l2_grade_dist" "conda activate reasoners" C-m
# tmux send-keys -t "maze_l2_grade_dist" "CUDA_VISIBLE_DEVICES=2 ./launch_maze.sh l2" C-m

# tmux new-session -t "maze_lm_grade_dist" -d -s
# tmux send-keys -t "maze_lm_grade_dist" "conda activate reasoners" C-m
# tmux send-keys -t "maze_lm_grade_dist" "CUDA_VISIBLE_DEVICES=1 ./launch_maze.sh lm" C-m

# sample="optimal_dist"
# tmux new-session -t "sokoban_lm_long_$sample-lr" -d -s
# tmux send-keys -t "sokoban_lm_long_$sample-lr" "conda activate reasoners" C-m
# tmux send-keys -t "sokoban_lm_long_$sample-lr" "CUDA_VISIBLE_DEVICES=0 ./launch_sokoban.sh lm $sample" C-m

# sample="optimal"
# tmux new-session -t "sokoban_lm_long_$sample-lr" -d -s
# tmux send-keys -t "sokoban_lm_long_$sample-lr" "conda activate reasoners" C-m
# tmux send-keys -t "sokoban_lm_long_$sample-lr" "CUDA_VISIBLE_DEVICES=3 ./launch_sokoban.sh lm $sample" C-m

# sample="optimal_easy_med"
# tmux new-session -t "maze_multipath_$sample" -d -s
# tmux send-keys -t "maze_multipath_$sample" "conda activate reasoners" C-m
# tmux send-keys -t "maze_multipath_$sample" "CUDA_VISIBLE_DEVICES=2 ./launch_maze.sh $sample" C-m

# sample="optimal_med_hard"
# tmux new-session -t "maze_multipath_$sample" -d -s
# tmux send-keys -t "maze_multipath_$sample" "conda activate reasoners" C-m
# tmux send-keys -t "maze_multipath_$sample" "CUDA_VISIBLE_DEVICES=1 ./launch_maze.sh $sample" C-m

# sample="optimal_easy_hard"
# tmux new-session -t "maze_multipath_$sample" -d -s
# tmux send-keys -t "maze_multipath_$sample" "conda activate reasoners" C-m
# tmux send-keys -t "maze_multipath_$sample" "CUDA_VISIBLE_DEVICES=3 ./launch_maze.sh $sample" C-m


sample="optimal"
sample_name="optimal"
tmux new-session -t "maze_lm_multipath_small_$sample_name" -d -s
tmux send-keys -t "maze_lm_multipath_small_$sample_name" "conda activate reasoners" C-m
tmux send-keys -t "maze_lm_multipath_small_$sample_name" "CUDA_VISIBLE_DEVICES=1 ./launch_maze.sh lm $sample" C-m

tmux new-session -t "maze_l2_multipath_small_$sample_name" -d -s
tmux send-keys -t "maze_l2_multipath_small_$sample_name" "conda activate reasoners" C-m
tmux send-keys -t "maze_l2_multipath_small_$sample_name" "CUDA_VISIBLE_DEVICES=3 ./launch_maze.sh l2 $sample" C-m

# tmux new-session -t "sokoban_lm_long_$sample-" -d -s
# tmux send-keys -t "sokoban_lm_long_$sample-" "conda activate reasoners" C-m
# tmux send-keys -t "sokoban_lm_long_$sample-" "CUDA_VISIBLE_DEVICES=2 ./launch_sokoban.sh lm $sample" C-m

# tmux new-session -t "sokoban_l2_long_$sample-" -d -s
# tmux send-keys -t "sokoban_l2_long_$sample-" "conda activate reasoners" C-m
# tmux send-keys -t "sokoban_l2_long_$sample-" "CUDA_VISIBLE_DEVICES=3 ./launch_sokoban.sh l2 $sample" C-m
