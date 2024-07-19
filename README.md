# Accelerating A* Search with Language Models

Official code implementation for the paper "[A Training Data Recipe to Accelerate A* Search with Language Models](https://arxiv.org/abs/2407.09985)".

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
```

### Data Generation
```bash
python main.py --domain $domain --dataset $dataset --create-data $parameters --job data_gen
```

The choices for ```--domain``` and ```--dataset``` can be found in the parser choices, in ```main.py```. 

```--create-data``` takes an ordered sequence of numbers, directly passed to ```create_$domain_dataset()``` in ```data/utils.py```.

To only sample new nodes, without creating a new puzzle file, pass 0 for ```num_train```, ```num_val``` and ```num_test``` in ```--create-data```. The number of sequences sampled per puzzle is controlled by ```--train-seqs``` and ```--val-seqs```. The sampling method is given by ```--sample``` and used in ```optimal_sample()``` in ```data/utils.py```. ```--dist-factor``` is the same as ```temperature``` in the paper. Also note that this code uses the terms *easy, medium, hard* in place of *end, middle, initial*, respectively.

### Training and Inference
```run.sh``` is used to run training and inference. Modifying the arguments in ```launch_$domain.sh``` is recommended before executing ```run.sh```. The only arguments that should need modification are ```--bs``` and ```--grad-step```.
