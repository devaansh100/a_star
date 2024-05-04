#!/bin/bash
sleep 1200
CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-grade --job 3k-optimal-3k-med --train-files alg_mazes_20_1000_optimal alg_mazes_20_2000_optimal alg_mazes_20_3000_optimal alg_mazes_20_4000_optimal_med alg_mazes_20_5000_optimal_med alg_mazes_20_6000_optimal_med --val-files alg_mazes_20_optimal --bs 128

CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-eval --job grade-6000-optimal_easy --train-files --lm model_best_test.pth --test-ilr test mazes_20
CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-eval --job grade-6000-optimal_easy --train-files --lm model_best_test.pth --test-ilr test mazes_30

CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-eval --job grade-6000-optimal_med --train-files --lm model_best_test.pth --test-ilr test mazes_20
CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-eval --job grade-6000-optimal_med --train-files --lm model_best_test.pth --test-ilr test mazes_30

CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-eval --job grade-6000-optimal_hard --train-files --lm model_best_test.pth --test-ilr test mazes_20
CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-eval --job grade-6000-optimal_hard --train-files --lm model_best_test.pth --test-ilr test mazes_30

CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-eval --job 3k-optimal-3k-med --train-files --lm model_best_test.pth --test-ilr test mazes_20
CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-eval --job 3k-optimal-3k-med --train-files --lm model_best_test.pth --test-ilr test mazes_30

# python main.py --domain maze --dataset maze-grade --job grade-6000-optimal --train-files alg_mazes_20_1000_optimal_r1 alg_mazes_20_2000_optimal_r1 alg_mazes_20_3000_optimal_r1 alg_mazes_20_4000_optimal_r1 alg_mazes_20_5000_optimal_r1 alg_mazes_20_6000_optimal_r1 --val-files alg_mazes_20_optimal_r1 --bs 128 --lm model_best_test.pth --prompt-file ../datasets/prompt_gb.txt --gb --rt