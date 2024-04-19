#!/bin/bash
sleep 12600
tmux send-keys -t maze_opt_dec4 "python main.py --domain maze --dataset maze-small --job opt_dec4 --train-files alg_mazes_opt_dec4 --bs 16" C-m
tmux send-keys -t maze_rand4_opt "CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-small --job rand4_opt --train-files alg_mazes_rand4_opt --bs 16" C-m
tmux send-keys -t maze_rand7_opt "CUDA_VISIBLE_DEVICES=3 python main.py --domain maze --dataset maze-small --job rand7_opt --train-files alg_mazes_rand7_opt --bs 16" C-m

sleep 14400
tmux send-keys -t maze_opt_dec7 "python main.py --domain maze --dataset maze-small --job opt_dec7 --train-files alg_mazes_opt_dec7 --bs 16" C-m
tmux send-keys -t maze_opt_dec10 "CUDA_VISIBLE_DEVICES=1 python main.py --domain maze --dataset maze-small --job opt_dec10 --train-files alg_mazes_opt_dec10 --bs 16" C-m
tmux send-keys -t maze_rand10_opt "CUDA_VISIBLE_DEVICES=3 python main.py --domain maze --dataset maze-small --job rand10_opt --train-files alg_mazes_rand10_opt --bs 16" C-m
# tmux send-keys -t maze_deception "python main.py --domain maze --dataset maze-small --train-files alg_mazes_deception --job deception"
# tmux send-keys -t maze_random "python main.py --domain maze --dataset maze-small --train-files alg_mazes_random --job random"

# python main.py --create-data 0 0 0 2 --sample opt_dec5
# python main.py --domain sokoban --dataset boxoban-small --train-files alg_sokoban_2_rand10_opt --job rand10_opt-small 
# ./run_length_gen.sh rand10_opt-small
# python main.py --domain sokoban --dataset boxoban-small --job rand1_opt-small --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr val sokoban_2
# python main.py --domain sokoban --dataset boxoban-small --job rand4_opt-small --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr val sokoban_2
# python main.py --domain sokoban --dataset boxoban-small --job rand_opt-small --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr val sokoban_2
# python main.py --domain sokoban --dataset boxoban-small --job rand10_opt-small --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr val sokoban_2
# --lm model_best_test.pth --test-ilr test mazes