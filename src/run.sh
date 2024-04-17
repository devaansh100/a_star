#!/bin/bash
# tmux send-keys -t data-optimal "python main.py --domain sokoban --dataset boxoban-small --train-files alg_sokoban_2_optimal" C-m
# python main.py --create-data 0 0 0 2 --sample opt_dec5
# python main.py --domain sokoban --dataset boxoban-small --train-files alg_sokoban_2_rand10_opt --job rand10_opt-small 
# ./run_length_gen.sh rand10_opt-small
python main.py --domain sokoban --dataset boxoban-small --job rand1_opt-small --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr val sokoban_2
python main.py --domain sokoban --dataset boxoban-small --job rand4_opt-small --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr val sokoban_2
python main.py --domain sokoban --dataset boxoban-small --job rand_opt-small --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr val sokoban_2
python main.py --domain sokoban --dataset boxoban-small --job rand10_opt-small --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr val sokoban_2