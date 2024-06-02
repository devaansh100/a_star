#!/bin/bash
if [ $1 = "lm" ]; then
    python main.py --domain maze --dataset maze-multipath-small --job multipath-small-$2 --train-files alg_mazes_20_$2 --val-files alg_mazes_20_$2
    python main.py --domain maze --dataset maze-multipath-eval --job multipath-small-$2 --lm model_best_test.pth --test-ilr test mazes_20
    python main.py --domain maze --dataset maze-multipath-eval --job multipath-small-$2 --lm model_best_test.pth --test-ilr test mazes_30
fi

if [ $1 = "l2" ]; then
    python main.py --domain maze --dataset maze-multipath-small --job multipath-small-$2 --train-files alg_mazes_20_$2 --val-files alg_mazes_20_$2 --loss l2
    python main.py --domain maze --dataset maze-multipath-eval --job multipath-small-$2 --lm model_best_test_l2_1.pth --test-ilr test mazes_20 --loss l2
    python main.py --domain maze --dataset maze-multipath-eval --job multipath-small-$2 --lm model_best_test_l2_1.pth --test-ilr test mazes_30 --loss l2
fi

# python main.py --domain maze --dataset maze-multipath --job multipath-6000-$1 --train-files alg_mazes_20_$1 --val-files alg_mazes_20_$1
# python main.py --domain maze --dataset maze-multipath --job multipath-6000-$1 --train-files alg_mazes_20_$1 --val-files alg_mazes_20_$1 --loss l2
# echo "LM IID eval"
# python main.py --domain maze --dataset maze-multipath-eval --job multipath-6000-$1 --lm model_best_test.pth --test-ilr test mazes_20
# echo "LM OOD eval"
# python main.py --domain maze --dataset maze-multipath-eval --job multipath-6000-$1 --lm model_best_test.pth --test-ilr test mazes_30
# echo "L2 IID eval"
# python main.py --domain maze --dataset maze-multipath-eval --job multipath-6000-$1 --lm model_best_test_l2_1.pth --test-ilr test mazes_20 --loss l2
# echo "L2 OOD eval"
# python main.py --domain maze --dataset maze-multipath-eval --job multipath-6000-$1 --lm model_best_test_l2_1.pth --test-ilr test mazes_30 --loss l2
