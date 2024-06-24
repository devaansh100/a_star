#!/bin/bash
if [ $1 = "lm" ]; then
    python main.py --domain maze --dataset maze-multipath-small --job multipath-small-$2 --train-files alg_mazes_20_$2_16 --val-files alg_mazes_20_$2_10 --base-model $3
    python main.py --domain maze --dataset maze-multipath-eval --job multipath-small-$2 --lm model_best_test.pth --test-ilr test mazes_20 --base-model $3
    python main.py --domain maze --dataset maze-multipath-eval --job multipath-small-$2 --lm model_best_test.pth --test-ilr test mazes_30 --base-model $3
fi

if [ $1 = "l2" ]; then
    python main.py --domain maze --dataset maze-multipath-small --job multipath-small-$2 --train-files alg_mazes_20_$2_16 --val-files alg_mazes_20_$2_10 --loss l2  --base-model $3
    python main.py --domain maze --dataset maze-multipath-eval --job multipath-small-$2 --lm model_best_test_l2_1.pth --test-ilr test mazes_20 --loss l2 --base-model $3
    python main.py --domain maze --dataset maze-multipath-eval --job multipath-small-$2 --lm model_best_test_l2_1.pth --test-ilr test mazes_30 --loss l2 --base-model $3
fi