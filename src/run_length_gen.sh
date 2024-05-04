#!/bin/bash
echo "1000 iid"
python main.py --domain sokoban --dataset boxoban-eval --job grade-1000-$1 --train-files --lm model_best_test.pth --test-ilr test sokoban_7k_2
echo "2000 iid"
python main.py --domain sokoban --dataset boxoban-eval --job grade-2000-$1 --train-files --lm model_best_test.pth --test-ilr test sokoban_7k_2
echo "3000 iid"
python main.py --domain sokoban --dataset boxoban-eval --job grade-3000-$1 --train-files --lm model_best_test.pth --test-ilr test sokoban_7k_2
echo "4000 iid"
python main.py --domain sokoban --dataset boxoban-eval --job grade-4000-$1 --train-files --lm model_best_test.pth --test-ilr test sokoban_7k_2
echo "5000 iid"
python main.py --domain sokoban --dataset boxoban-eval --job grade-5000-$1 --train-files --lm model_best_test.pth --test-ilr test sokoban_7k_2
echo "6000 iid"
python main.py --domain sokoban --dataset boxoban-eval --job grade-6000-$1 --train-files --lm model_best_test.pth --test-ilr test sokoban_7k_2

echo "1000 ood"
python main.py --domain sokoban --dataset boxoban-eval --job grade-1000-$1 --train-files --lm model_best_test.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4
echo "2000 ood"
python main.py --domain sokoban --dataset boxoban-eval --job grade-2000-$1 --train-files --lm model_best_test.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4
echo "3000 ood"
python main.py --domain sokoban --dataset boxoban-eval --job grade-3000-$1 --train-files --lm model_best_test.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4
echo "4000 ood"
python main.py --domain sokoban --dataset boxoban-eval --job grade-4000-$1 --train-files --lm model_best_test.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4
echo "5000 ood"
python main.py --domain sokoban --dataset boxoban-eval --job grade-5000-$1 --train-files --lm model_best_test.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4
echo "6000 ood"
python main.py --domain sokoban --dataset boxoban-eval --job grade-6000-$1 --train-files --lm model_best_test.pth --test-ilr test sokoban_2 sokoban_3 sokoban_4 sokoban_7k_4

# echo "sokoban_2"
# python main.py --domain sokoban --dataset boxoban-length-gen --job $1 --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr test sokoban_2
# echo "sokoban_3"
# python main.py --domain sokoban --dataset boxoban-length-gen --job $1 --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr test sokoban_3
# echo "sokoban_4"
# python main.py --domain sokoban --dataset boxoban-length-gen --job $1 --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr test sokoban_4
# echo "sokoban_7k_4"
# python main.py --domain sokoban --dataset boxoban-length-gen --job $1 --train-files --base-model code-t5 --lm model_best_test.pth --test-ilr test sokoban_7k_4
# echo "mazes_20"
# tmux new-session -t "sokoban_$1" -d -s
# tmux send-keys -t "sokoban_6000_iid_eval_$1" "conda activate reasoners" C-m
# tmux send-keys -t "sokoban_6000_iid_eval_$1" "CUDA_VISIBLE_DEVICES=0 python main.py --domain maze --dataset maze-eval --job grade-6000-$1 --train-files --lm model_best_test.pth --test-ilr test mazes_20" C-m
# echo "out of domain"
# tmux new-session -t "sokoban_6000_ood_eval_$1" -d -s
# tmux send-keys -t "sokoban_6000_ood_eval_$1" "conda activate reasoners" C-m
# tmux send-keys -t "sokoban_6000_ood_eval_$1" "CUDA_VISIBLE_DEVICES=3 python main.py --domain maze --dataset maze-eval --job grade-6000-$1 --train-files --lm model_best_test.pth --test-ilr test mazes_30" C-m
# echo "mazes_40"
# python main.py --domain maze --dataset maze-eval --job grade-6000-$1 --train-files --lm model_best_test.pth --test-ilr test mazes_40

# python main.py --domain maze --dataset maze-eval --job large-opt_dec4 --train-files --lm model_best_test.pth --test-ilr test mazes_20
# python main.py --domain maze --dataset maze-eval --job large-opt_dec4 --train-files --lm model_best_test.pth --test-ilr test mazes_30
# python main.py --domain maze --dataset maze-eval --job large-opt_dec4 --train-files --lm model_best_test.pth --test-ilr test mazes_40

# python main.py --domain maze --dataset maze-eval --job large-rand4_opt --train-files --lm model_best_test.pth --test-ilr test mazes_20
# python main.py --domain maze --dataset maze-eval --job large-rand4_opt --train-files --lm model_best_test.pth --test-ilr test mazes_30
# python main.py --domain maze --dataset maze-eval --job large-rand4_opt --train-files --lm model_best_test.pth --test-ilr test mazes_40
# maze
# echo "mazes_10_1000"
# python main.py --domain maze --dataset maze-eval --job grade-1000-$1 --train-files --lm model_best_test.pth --test-ilr test mazes_10

# echo "mazes_10_2000"
# python main.py --domain maze --dataset maze-eval --job grade-2000-$1 --train-files --lm model_best_test.pth --test-ilr test mazes_10

# echo "mazes_10_3000"
# python main.py --domain maze --dataset maze-eval --job grade-3000-$1 --train-files --lm model_best_test.pth --test-ilr test mazes_10

# echo "mazes_10_4000"
# python main.py --domain maze --dataset maze-eval --job grade-4000-$1 --train-files --lm model_best_test.pth --test-ilr test mazes_10

# echo "mazes_10_5000"
# python main.py --domain maze --dataset maze-eval --job grade-5000-$1 --train-files --lm model_best_test.pth --test-ilr test mazes_10

# echo "mazes_10_6000"
# python main.py --domain maze --dataset maze-eval --job grade-6000-$1 --train-files --lm model_best_test.pth --test-ilr test mazes_10

# echo "all_files_1000"
# python main.py --domain maze --dataset maze-eval --job grade-1000-$1 --train-files --lm model_best_test.pth --test-ilr test mazes_20 mazes_30 mazes_40

# echo "all_files_2000"
# python main.py --domain maze --dataset maze-eval --job grade-2000-$1 --train-files --lm model_best_test.pth --test-ilr test mazes_20 mazes_30 mazes_40

# echo "all_files_3000"
# python main.py --domain maze --dataset maze-eval --job grade-3000-$1 --train-files --lm model_best_test.pth --test-ilr test mazes_20 mazes_30 mazes_40

# echo "all_files_4000"
# python main.py --domain maze --dataset maze-eval --job grade-4000-$1 --train-files --lm model_best_test.pth --test-ilr test mazes_20 mazes_30 mazes_40

# echo "all_files_5000"
# python main.py --domain maze --dataset maze-eval --job grade-5000-$1 --train-files --lm model_best_test.pth --test-ilr test mazes_20 mazes_30 mazes_40

# echo "all_files_6000"
# python main.py --domain maze --dataset maze-eval --job grade-6000-$1 --train-files --lm model_best_test.pth --test-ilr test mazes_20 mazes_30 mazes_40