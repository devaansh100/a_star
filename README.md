# Accelerating A* Search with Language Models

Official code implementation for the paper "[A Training Data Recipe to Accelerate A* Search with Language Models](https://aclanthology.org/2024.findings-emnlp.391/)", accepted at Findings of EMNLP 2024.

[arxiv](https://arxiv.org/abs/2407.09985)

## Environment

```bash
conda env create --file environment.yml
conda activate a_star
```

## Data

### Datasets used in the paper
```bash
cd datasets
unzip maze.zip
unzip boxoban-fin.zip
unzip boxoban-fin-eval.zip
unzip stp-alph/test.zip
unzip stp-alph/val.zip
unzip semdedup.zip # For the coresets created by semdedup
```

### Data Generation
Use the following command to generate new puzzles/sequences for training.
```bash
python main.py --domain $domain --dataset $dataset --create-data $parameters --job data_gen
```

For sokoban and maze, additional libraries need to be cloned. Refer [below](#additional-libraries)

The choices for ```--domain``` and ```--dataset``` can be found in the parser choices, in ```main.py```. 

```--create-data``` takes an ordered sequence of numbers, directly passed to ```create_$domain_dataset()``` in ```data/utils.py```.

To only sample new nodes, without creating a new puzzle file, pass 0 for ```num_train```, ```num_val``` and ```num_test``` in ```--create-data```. The number of sequences sampled per puzzle is controlled by ```--train-seqs``` and ```--val-seqs```. The sampling method is given by ```--sample``` and used in ```optimal_sample()``` in ```data/utils.py```. ```--dist-factor``` is the same as ```temperature``` in the paper. Also note that this code uses the terms *easy, medium, hard* in place of *end, middle, initial*, respectively.

### Training and Inference
```run.sh``` is used to run training and inference. Modifying the arguments in ```launch_$domain.sh``` is recommended before executing ```run.sh```. The only arguments that should need modification are ```--bs``` and ```--grad-step```.

### Additional Libraries

#### Maze
For generation of mazes, we use the mazelib library. Execute the following command in the ```src/data``` directory.
```
git clone https://github.com/john-science/mazelib.git
```

#### Sokoban
For sokoban, the original boxoban dataset will need to be installed. Execute the following command in the ```datasets``` directory.
```
git clone https://github.com/google-deepmind/boxoban-levels.git
```


If the code and/or method was useful to your work, consider citing us:
```
@inproceedings{gupta-li-2024-training,
    title = "A Training Data Recipe to Accelerate A* Search with Language Models",
    author = "Gupta, Devaansh and Li, Boyang",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    year = "2024",
}
```
