#!/bin/bash
# if [ $1 = "lm" ]; then
#     python main.py --domain sokoban --dataset boxoban-grade --job grade-2000-med_hard --train-files alg_sokoban_2_1000_optimal_med_hard alg_sokoban_2_2000_optimal_med_hard --val-files alg_sokoban_2_optimal_med_hard
#     python main.py --domain sokoban --dataset boxoban-eval --job grade-2000-med_hard --lm model_best_test.pth --test-ilr test sokoban_7k_2
#     python main.py --domain sokoban --dataset boxoban-eval --job grade-2000-med_hard --lm model_best_test.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4
# fi

# if [ $1 = "l2" ]; then
#     python main.py --domain sokoban --dataset boxoban-grade --job grade-2000-med_hard --train-files alg_sokoban_2_1000_optimal_med_hard alg_sokoban_2_2000_optimal_med_hard --val-files alg_sokoban_2_optimal_med_hard --loss l2 --num-heads 1
#     python main.py --domain sokoban --dataset boxoban-eval --job grade-2000-med_hard --lm model_best_test_l2_1.pth --test-ilr test sokoban_7k_2 --loss l2
#     python main.py --domain sokoban --dataset boxoban-eval --job grade-2000-med_hard --lm model_best_test_l2_1.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4 --loss l2
# fi

if [ $1 = "lm" ]; then
    python main.py --domain sokoban --dataset boxoban-long --job long-1000-$2 --train-files alg_sokoban_2_$2 --val-files alg_sokoban_2_$2
    python main.py --domain sokoban --dataset boxoban-eval --job long-1000-$2 --lm model_best_test.pth --test-ilr test sokoban_7k_2
    python main.py --domain sokoban --dataset boxoban-eval --job long-1000-$2 --lm model_best_test.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4
fi

if [ $1 = "l2" ]; then
    python main.py --domain sokoban --dataset boxoban-long --job long-1000-$2 --train-files alg_sokoban_2_$2 --val-files alg_sokoban_2_$2 --loss l2
    python main.py --domain sokoban --dataset boxoban-eval --job long-1000-$2 --lm model_best_test_l2_1.pth --test-ilr test sokoban_7k_2 --loss l2
    python main.py --domain sokoban --dataset boxoban-eval --job long-1000-$2 --lm model_best_test_l2_1.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4 --loss l2
fi