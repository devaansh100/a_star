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

tmux new-session -t "sokoban_lm_grade_optimal" -d -s
tmux send-keys -t "sokoban_lm_grade_optimal" "conda activate reasoners" C-m
tmux send-keys -t "sokoban_lm_grade_optimal" "CUDA_VISIBLE_DEVICES=0 ./launch_sokoban.sh lm" C-m

tmux new-session -t "sokoban_l2_grade_optimal" -d -s
tmux send-keys -t "sokoban_l2_grade_optimal" "conda activate reasoners" C-m
tmux send-keys -t "sokoban_l2_grade_optimal" "CUDA_VISIBLE_DEVICES=1 ./launch_sokoban.sh l2" C-m

tmux new-session -t "maze_l2_grade_optimal" -d -s
tmux send-keys -t "maze_l2_grade_optimal" "conda activate reasoners" C-m
tmux send-keys -t "maze_l2_grade_optimal" "CUDA_VISIBLE_DEVICES=2 ./launch_maze.sh l2" C-m

tmux new-session -t "maze_lm_grade_optimal" -d -s
tmux send-keys -t "maze_lm_grade_optimal" "conda activate reasoners" C-m
tmux send-keys -t "maze_lm_grade_optimal" "CUDA_VISIBLE_DEVICES=3 ./launch_maze.sh lm" C-m
