#!/bin/bash
if [ $1 = "lm" ]; then
    python main.py --domain maze --dataset maze-small --job maze-small-easy --train-files alg_mazes_20_optimal_easy --val-files alg_mazes_20_optimal_easy
    # python main.py --domain maze --dataset maze-small --job maze-small-easy --train-files alg_mazes_20_optimal --val-files alg_mazes_20_optimal --lm model_best_test_20.pth --create-gb-data r1_weak_lm --test
    # python main.py --domain maze --dataset maze-small --job maze-small-easy --train-files alg_mazes_20_optimal alg_mazes_20_optimal_r1_weak_lm --val-files alg_mazes_20_optimal --suffix lm_ss --num-epochs 40
    # echo "Evaluating base model"
    python main.py --domain maze --dataset maze-eval --job maze-small-easy --lm model_best_test.pth --test-ilr test mazes_20
    python main.py --domain maze --dataset maze-eval --job maze-small-easy --lm model_best_test.pth --test-ilr test mazes_30
    # echo "Evaluating GB"
    # python main.py --domain maze --dataset maze-eval --job maze-small-easy --lm model_best_test_lm_ss.pth --test-ilr test mazes_20
    # python main.py --domain maze --dataset maze-eval --job maze-small-easy --lm model_best_test_lm_ss.pth --test-ilr test mazes_30
fi

if [ $1 = "l2" ]; then
    python main.py --domain maze --dataset maze-small --job maze-small-easy --train-files alg_mazes_20_optimal_easy --val-files alg_mazes_20_optimal_easy --loss l2 --num-heads 1
    # python main.py --domain maze --dataset maze-small --job maze-small-easy --train-files alg_mazes_20_optimal --val-files alg_mazes_20_optimal --lm model_best_test_l2_1.pth --create-gb-data r1_weak_l2 --test --loss l2
    # python main.py --domain maze --dataset maze-small --job maze-small-easy --train-files alg_mazes_20_optimal_r1_weak_l2 --val-files alg_mazes_20_optimal --suffix l2_ss --num-epochs 10 --loss l2 --lm model_best_test_l2_1.pth --rt
    # echo "Evaluating base model"
    python main.py --domain maze --dataset maze-eval --job maze-small-easy --lm model_best_test_l2_1.pth --test-ilr test mazes_20 --loss l2
    python main.py --domain maze --dataset maze-eval --job maze-small-easy --lm model_best_test_l2_1.pth --test-ilr test mazes_30 --loss l2
    # echo "Evaluating GB"
    # python main.py --domain maze --dataset maze-eval --job maze-small-easy --lm model_best_test_l2_ss.pth --test-ilr test mazes_20 --loss l2
    # python main.py --domain maze --dataset maze-eval --job maze-small-easy --lm model_best_test_l2_ss.pth --test-ilr test mazes_30 --loss l2
fi