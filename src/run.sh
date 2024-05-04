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

sleep 5400
tmux send-keys -t "sokoban_l2_long" "CUDA_VISIBLE_DEVICES=0 python main.py --domain sokoban --dataset boxoban-eval --job grade-1000-optimal_hard --train-files --lm model_best_test.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4" C-m
# tmux send-keys -t "maze-6000-l2-bs" "CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-grade --job grade-6000-optimal --train-files alg_mazes_20_1000_optimal alg_mazes_20_2000_optimal alg_mazes_20_3000_optimal alg_mazes_20_4000_optimal alg_mazes_20_5000_optimal alg_mazes_20_6000_optimal --val-files alg_mazes_20_optimal --loss l2 --num-heads 2 --lm model_best_test_l2_1.pth --rt" C-m
# tmux send-keys -t "maze-6000-l2-bs" "CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-eval --job grade-6000-optimal --lm model_best_test_l2_1.pth --loss l2 --test-ilr test mazes_20" C-m
# tmux send-keys -t "maze-6000-l2-bs" "CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-eval --job grade-6000-optimal --lm model_best_test_l2_1.pth --loss l2 --test-ilr test mazes_30" C-m
# tmux send-keys -t "maze-6000-l2-bs" "CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-eval --job grade-6000-optimal --lm model_best_test_l2_2.pth --loss l2 --num-heads 2 --test-ilr test mazes_20" C-m
# tmux send-keys -t "maze-6000-l2-bs" "CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-eval --job grade-6000-optimal --lm model_best_test_l2_2.pth --loss l2 --num-heads 2 --test-ilr test mazes_30" C-m


# for sample in "${samples[@]}"; do
#     tmux new-session -t "sokoban_1000_$sample" -d -s
#     tmux send-keys -t "sokoban_1000_$sample" "conda activate reasoners" C-m
#     tmux send-keys -t "sokoban_1000_$sample" "CUDA_VISIBLE_DEVICES=0 python main.py --domain sokoban --dataset boxoban-grade --job grade-1000-$sample --train-files alg_sokoban_2_1000_$sample --val-files alg_sokoban_2_optimal" C-m

#     tmux new-session -t "sokoban_2000_$sample" -d -s
#     tmux send-keys -t "sokoban_2000_$sample" "conda activate reasoners" C-m
#     tmux send-keys -t "sokoban_2000_$sample" "CUDA_VISIBLE_DEVICES=1 python main.py --domain sokoban --dataset boxoban-grade --job grade-2000-$sample --train-files alg_sokoban_2_1000_$sample alg_sokoban_2_2000_$sample --val-files alg_sokoban_2_optimal" C-m

#     tmux new-session -t "sokoban_3000_$sample" -d -s
#     tmux send-keys -t "sokoban_3000_$sample" "conda activate reasoners" C-m
#     tmux send-keys -t "sokoban_3000_$sample" "CUDA_VISIBLE_DEVICES=3 python main.py --domain sokoban --dataset boxoban-grade --job grade-3000-$sample --train-files alg_sokoban_2_1000_$sample alg_sokoban_2_2000_$sample alg_sokoban_2_3000_$sample --val-files alg_sokoban_2_optimal" C-m
#     sleep 2000

#     tmux new-session -t "sokoban_4000_$sample" -d -s
#     tmux send-keys -t "sokoban_4000_$sample" "conda activate reasoners" C-m
#     tmux send-keys -t "sokoban_4000_$sample" "CUDA_VISIBLE_DEVICES=0 python main.py --domain sokoban --dataset boxoban-grade --job grade-4000-$sample --train-files alg_sokoban_2_1000_$sample alg_sokoban_2_2000_$sample alg_sokoban_2_3000_$sample alg_sokoban_2_4000_$sample --val-files alg_sokoban_2_optimal" C-m

#     tmux new-session -t "sokoban_5000_$sample" -d -s
#     tmux send-keys -t "sokoban_5000_$sample" "conda activate reasoners" C-m
#     tmux send-keys -t "sokoban_5000_$sample" "CUDA_VISIBLE_DEVICES=1 python main.py --domain sokoban --dataset boxoban-grade --job grade-5000-$sample --train-files alg_sokoban_2_1000_$sample alg_sokoban_2_2000_$sample alg_sokoban_2_3000_$sample alg_sokoban_2_4000_$sample alg_sokoban_2_5000_$sample --val-files alg_sokoban_2_optimal" C-m

#     tmux new-session -t "sokoban_6000_$sample" -d -s
#     tmux send-keys -t "sokoban_6000_$sample" "conda activate reasoners" C-m
#     tmux send-keys -t "sokoban_6000_$sample" "CUDA_VISIBLE_DEVICES=3 python main.py --domain sokoban --dataset boxoban-grade --job grade-6000-$sample --train-files alg_sokoban_2_1000_$sample alg_sokoban_2_2000_$sample alg_sokoban_2_3000_$sample alg_sokoban_2_4000_$sample alg_sokoban_2_5000_$sample alg_sokoban_2_6000_$sample --val-files alg_sokoban_2_optimal" C-m
#     sleep 3780
# done

# tmux new-session -t "sokoban_optimal_easy_eval" -d -s
# tmux send-keys -t "sokoban_optimal_easy_eval" "conda activate reasoners" C-m
# tmux send-keys -t "sokoban_optimal_easy_eval" "CUDA_VISIBLE_DEVICES=0 ./run_length_gen.sh optimal_easy" C-m

# tmux new-session -t "sokoban_optimal_med_eval" -d -s
# tmux send-keys -t "sokoban_optimal_med_eval" "conda activate reasoners" C-m
# tmux send-keys -t "sokoban_optimal_med_eval" "CUDA_VISIBLE_DEVICES=1 ./run_length_gen.sh optimal_med" C-m

# tmux new-session -t "sokoban_optimal_hard_eval" -d -s
# tmux send-keys -t "sokoban_optimal_hard_eval" "conda activate reasoners" C-m
# tmux send-keys -t "sokoban_optimal_hard_eval" "CUDA_VISIBLE_DEVICES=3 ./run_length_gen.sh optimal_hard" C-m