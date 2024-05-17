#!/bin/bash
if [ $1 = "lm" ]; then
    python main.py --domain sokoban --dataset boxoban-small --job boxoban-small_easy --train-files alg_sokoban_2_optimal_easy --val-files alg_sokoban_2_optimal_easy
    # python main.py --domain sokoban --dataset boxoban-small --job boxoban-small_easy --train-files alg_sokoban_2_optimal --val-files alg_sokoban_2_optimal --lm model_best_test.pth --create-gb-data r1_weak_lm --test
    # python main.py --domain sokoban --dataset boxoban-small --job boxoban-small_easy --train-files alg_sokoban_2_optimal alg_sokoban_2_optimal_r1_weak_lm --val-files alg_sokoban_2_optimal --suffix lm_ss --num-epochs 40
    # echo "Evaluating base model"
    python main.py --domain sokoban --dataset boxoban-eval --job boxoban-small_easy --lm model_best_test.pth --test-ilr test sokoban_7k_2
    python main.py --domain sokoban --dataset boxoban-eval --job boxoban-small_easy --lm model_best_test.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4
    # echo "Evaluating GB"
    # python main.py --domain sokoban --dataset boxoban-eval --job boxoban-small_easy --lm model_best_test_lm_ss.pth --test-ilr test sokoban_7k_2
    # python main.py --domain sokoban --dataset boxoban-eval --job boxoban-small_easy --lm model_best_test_lm_ss.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4
fi

if [ $1 = "l2" ]; then
    python main.py --domain sokoban --dataset boxoban-small --job boxoban-small_easy --train-files alg_sokoban_2_optimal_easy --val-files alg_sokoban_2_optimal_easy --loss l2 --num-heads 1
    # python main.py --domain sokoban --dataset boxoban-small --job boxoban-small_easy --train-files alg_sokoban_2_optimal --val-files alg_sokoban_2_optimal --loss l2 --lm model_best_test_l2_2.pth --create-gb-data r1_weak_l2 --test
    # python main.py --domain sokoban --dataset boxoban-small --job boxoban-small_easy --train-files alg_sokoban_2_optimal_r1_weak_l2 --val-files alg_sokoban_2_optimal --loss l2 --suffix l2_ss --num-epochs 20 --lm model_best_test_l2_1.pth --rt
    # echo "Evaluating base model"
    python main.py --domain sokoban --dataset boxoban-eval --job boxoban-small_easy --lm model_best_test_l2_1.pth --test-ilr test sokoban_7k_2 --loss l2
    python main.py --domain sokoban --dataset boxoban-eval --job boxoban-small_easy --lm model_best_test_l2_1.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4 --loss l2
    # echo "Evaluating GB"
    # python main.py --domain sokoban --dataset boxoban-eval --job boxoban-small_easy --lm model_best_test_l2_ss.pth --test-ilr test sokoban_7k_2 --loss l2
    # python main.py --domain sokoban --dataset boxoban-eval --job boxoban-small_easy --lm model_best_test_l2_ss.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4 --loss l2
fi