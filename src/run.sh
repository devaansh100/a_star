#!/bin/bash
# echo "sokoban_2"
# CUDA_VISIBLE_DEVICES=3 python main.py --domain sokoban --dataset boxoban-length-gen --job fixed_data --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr test sokoban_2
# echo "sokoban_3"
# CUDA_VISIBLE_DEVICES=3 python main.py --domain sokoban --dataset boxoban-length-gen --job fixed_data --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr test sokoban_3
# echo "sokoban_4"
# CUDA_VISIBLE_DEVICES=3 python main.py --domain sokoban --dataset boxoban-length-gen --job fixed_data --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr test sokoban_4
# echo "sokoban_7k_4"
# CUDA_VISIBLE_DEVICES=3 python main.py --domain sokoban --dataset boxoban-length-gen --job fixed_data --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr test sokoban_7k_4

python main.py --domain sokoban --dataset boxoban-astar-rand --job random --train-files alg_sokoban_2
python main.py --domain sokoban --dataset boxoban-astar-rand --job random --train-files alg_sokoban_2 --lm model_best_test.pth --test-ilr test sokoban_2