#!/bin/bash
python main.py --domain stp --dataset stp-data --job data-gen --create-data 1000 1000 500 3
mv ../datasets/stp-data/test/* ../datasets/stp-eval/test
rmdir ../datasets/stp-data/test

if [ $1 = "lm" ]; then
    python main.py --domain stp --dataset stp-data --job stp-$2 --train-files alg_stp_3_$2_8 --val-files alg_stp_3_$2_5 --base-model $3
    python main.py --domain stp --dataset stp-eval --job stp-$2 --lm model_best_test.pth --test-ilr test stp_3 --base-model $3
    # python main.py --domain stp --dataset stp-eval --job stp-$2 --lm model_best_test.pth --test-ilr test stp_4 stp_5 --base-model $3
fi

if [ $1 = "l2" ]; then
    python main.py --domain stp --dataset stp-data --job stp-$2 --train-files alg_stp_3_$2_8 --val-files alg_stp_3_$2_5 --loss l2 --base-model $3
    python main.py --domain stp --dataset stp-eval --job stp-$2 --lm model_best_test_l2_1.pth --test-ilr test stp_3 --loss l2 --base-model $3
    # python main.py --domain stp --dataset stp-eval --job stp-$2 --lm model_best_test_l2_1.pth --test-ilr test stp_4 stp_5 --loss l2 --base-model $3
fi