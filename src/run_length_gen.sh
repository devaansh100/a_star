#!/bin/bash
echo "sokoban_7k_2"
python main.py --domain sokoban --dataset boxoban-small --job $1 --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr test sokoban_2
echo "all files"
python main.py --domain sokoban --dataset boxoban-length-gen --job $1 --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4
echo "sokoban_2"
python main.py --domain sokoban --dataset boxoban-length-gen --job $1 --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr test sokoban_2
echo "sokoban_3"
python main.py --domain sokoban --dataset boxoban-length-gen --job $1 --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr test sokoban_3
echo "sokoban_4"
python main.py --domain sokoban --dataset boxoban-length-gen --job $1 --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr test sokoban_4
echo "sokoban_7k_4"
python main.py --domain sokoban --dataset boxoban-length-gen --job $1 --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr test sokoban_7k_4

# python main.py --domain sokoban --dataset boxoban-astar-dec --job deception --train-files alg_sokoban_2
# python main.py --domain sokoban --dataset boxoban-astar-dec --job random --train-files alg_sokoban_2 --lm model_best_test.pth --test-ilr test sokoban_2