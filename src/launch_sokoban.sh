#!/bin/bash
if [ $1 = "lm" ]; then
    python main.py --domain sokoban --dataset boxoban-fin --job fin-1000-$2 --train-files alg_sokoban_2_$2_8 --val-files alg_sokoban_2_$2_5 --base-model $3
    python main.py --domain sokoban --dataset boxoban-fin-eval --job fin-1000-$2 --lm model_best_test.pth --test-ilr test sokoban_2_20_0 --base-model $3
    python main.py --domain sokoban --dataset boxoban-fin-eval --job fin-1000-$2 --lm model_best_test.pth --test-ilr test sokoban_2_20_7000 sokoban_3_20_0 sokoban_4_20_0 sokoban_3_20_7000 sokoban_4_20_7000 --base-model $3
fi

if [ $1 = "l2" ]; then
    python main.py --domain sokoban --dataset boxoban-fin --job fin-1000-$2 --train-files alg_sokoban_2_$2_8 --val-files alg_sokoban_2_$2_5 --loss l2 --base-model $3
    python main.py --domain sokoban --dataset boxoban-fin-eval --job fin-1000-$2 --lm model_best_test_l2_1.pth --test-ilr test sokoban_2_20_0 --loss l2 --base-model $3
    python main.py --domain sokoban --dataset boxoban-fin-eval --job fin-1000-$2 --lm model_best_test_l2_1.pth --test-ilr test sokoban_2_20_7000 sokoban_3_20_0 sokoban_4_20_0 sokoban_3_20_7000 sokoban_4_20_7000 --loss l2 --base-model $3
fi