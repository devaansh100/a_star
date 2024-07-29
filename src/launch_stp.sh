#!/bin/bash
if [ $1 = "lm" ]; then
    python main.py --domain stp --dataset stp-alph --job fin-stp-alph-$2 --train-files alg_stp_3_$2_8 --val-files alg_stp_3_$2_5 --base-model $3
    python main.py --domain stp --dataset stp-alph --job fin-stp-alph-$2 --lm model_best_test.pth --test-ilr test stp_3 --base-model $3 --train-files
    python main.py --domain stp --dataset stp-alph --job fin-stp-alph-$2 --lm model_best_test.pth --test-ilr test stp_4 stp_5 --base-model $3 --train-files
fi

if [ $1 = "l2" ]; then
    python main.py --domain stp --dataset stp-alph --job fin-stp-alph-$2 --train-files alg_stp_3_$2_8 --val-files alg_stp_3_$2_5 --loss l2 --base-model $3
    python main.py --domain stp --dataset stp-alph --job fin-stp-alph-$2 --lm model_best_test_l2_1.pth --test-ilr test stp_3 --loss l2 --base-model $3 --train-files
    python main.py --domain stp --dataset stp-alph --job fin-stp-alph-$2 --lm model_best_test_l2_1.pth --test-ilr test stp_4 stp_5 --loss l2 --base-model $3 --train-files
fi