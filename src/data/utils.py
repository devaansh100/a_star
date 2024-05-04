from tqdm import tqdm
import random
import os
import pickle as pkl
import numpy as np
from algorithm.utils import *
from algorithm import AStar_maze, AStar_sokoban
import glob
from transformers import AutoTokenizer
import torch
import sys
sys.path.append('data/mazelib')
from mazelib import Maze
from mazelib.generate.Prims import Prims

def generate_maze(width, height):
	m = Maze()
	m.generator = Prims(width, height)
	m.generate()
	return m

def create_maze_dataset(params, num_train, num_val, num_test, size):
	puzzles = []
	p_bar = tqdm(range(num_train + num_val + num_test))
	mazes = set()
	while len(puzzles) < num_train + num_val + num_test:
		m = generate_maze(size, size)
		iters = 10
		while iters > 0:
			m.generate_entrances(start_outer=False, end_outer=False)
			maze = convert_array_to_maze(m.grid, m.start, m.end)
			alg = AStar_maze(maze)
			if maze not in mazes:
				mazes.add(maze)
				alg.search()
				if alg.optimal_plan is not None:
					if len(alg.optimal_plan) > 2*size and len(alg.closed)/len(alg.optimal_plan) > 3:
						# print(maze)
						puzzles.append((maze, alg))
						p_bar.n += 1
						p_bar.refresh()
					iters = 0
			iters -= 1
	
	random.shuffle(puzzles)
	train, val, test = puzzles[:num_train], puzzles[num_train: num_train + num_val], puzzles[num_train + num_val:]
	print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
	os.makedirs(f'{params.data_dir}/{params.dataset}/train', exist_ok = True)
	os.makedirs(f'{params.data_dir}/{params.dataset}/val', exist_ok = True)
	os.makedirs(f'{params.data_dir}/{params.dataset}/test', exist_ok = True)
	
	if num_train > 0:
		# for i in range(1, 1 + len(train) // 1000):
		# 	with open(f'{params.data_dir}/{params.dataset}/train/mazes_{2*size}_{i * 1000}.txt', 'w') as f:
		# 		f.write(';'.join([x[0] for x in train[(i - 1) * 1000 : i * 1000]]))

		# 	with open(f'{params.data_dir}/{params.dataset}/train/alg_mazes_{2*size}_{i * 1000}.pkl', 'wb') as f:
		# 		pkl.dump([x[1] for x in train[(i - 1) * 1000 : i * 1000]], f)
		with open(f'{params.data_dir}/{params.dataset}/train/mazes_{2*size}.txt', 'w') as f:
			f.write(';'.join([x[0] for x in train]))
		
		with open(f'{params.data_dir}/{params.dataset}/train/alg_mazes_{2*size}.pkl', 'wb') as f:
			pkl.dump([x[1] for x in train], f)
	
	if num_val > 0:
		with open(f'{params.data_dir}/{params.dataset}/val/mazes_{2*size}.txt', 'w') as f:
			f.write(';'.join([x[0] for x in val]))
		
		with open(f'{params.data_dir}/{params.dataset}/val/alg_mazes_{2*size}.pkl', 'wb') as f:
			pkl.dump([x[1] for x in val], f)
	if num_test > 0:
		with open(f'{params.data_dir}/{params.dataset}/test/mazes_{2*size}.txt', 'w') as f:
			f.write(';'.join([x[0] for x in test]))
				
		with open(f'{params.data_dir}/{params.dataset}/test/alg_mazes_{2*size}.pkl', 'wb') as f:
			pkl.dump([x[1] for x in test], f)

def read_boxoban(params, split, hardness = 'unfiltered'):
	path = f'{params.data_dir}/boxoban-levels/{hardness}/{split}/'
	files = os.listdir(path)
	puzzles = []
	for file in files:
		with open(path + file) as f:
			f_puzzles = f.read()
		f_puzzles = f_puzzles.split(';')[1:]
		puzzles.extend(f_puzzles)
	return puzzles

def create_sokoban_dataset(params, num_train, num_val, num_test, subsample, terminate_after = 7000, min_iterations = 0, solver = None):
	solver = AStar_sokoban if solver is None else solver
	train = read_boxoban(params, 'train')
	val = read_boxoban(params, 'valid')
	test = read_boxoban(params, 'test')
	random.shuffle(train)
	random.shuffle(val)
	random.shuffle(test)
	for split, puzzles in zip(['train', 'val', 'test'], [train, val, test]):
		astar, algs = [], []
		if split == 'train':
			num_puzzles = num_train
		elif split == 'val':
			num_puzzles = num_val		
		else:
			num_puzzles = num_test
		
		if num_puzzles > 0:
			p_bar = tqdm(range(num_puzzles), desc = split)	
			while len(astar) < num_puzzles and len(puzzles) > 0:
				puzzle = puzzles.pop()
				puzzle = puzzle[puzzle.index('\n') + 1:].strip()
				alg = solver(puzzle, subsample = subsample, terminate_after = terminate_after)
				alg.search()
				if alg.optimal_plan is not None:
					if alg.iterations > min_iterations:
						puzzle = convert_array_to_sb(alg.puzzle, alg.docks, alg.initial_boxes, alg.initial_pos)
						astar.append(puzzle)
						algs.append(alg)
						p_bar.n += 1
						p_bar.refresh()
			print(f'{split}: {len(astar)}')
			path = f'{params.data_dir}/{params.dataset}/{split}/'
			os.makedirs(path, exist_ok = True)
			
			if split == 'train':
				for i in range(1, 1 + len(astar) // 1000):
					with open(path + f'sokoban_{subsample}_{i * 1000}.txt', 'w') as f:
						f.write(';'.join(astar[(i - 1) * 1000 : i * 1000]))

					with open(path + f'alg_sokoban_{subsample}_{i * 1000}.pkl', 'wb') as f:
						pkl.dump(algs[(i - 1) * 1000 : i * 1000], f)
			else:
				with open(path + f'sokoban_{subsample}.txt', 'w') as f:
					f.write(';'.join(astar))

				with open(path + f'alg_sokoban_{subsample}.pkl', 'wb') as f:
					pkl.dump(algs, f)

def annotate_deceptive_nodes(alg):
	for node in alg.closed:
		num_generations = float('Inf')
		while not node.is_optimal and num_generations > 0:
			node.deception_score += 1
			num_generations -= 1
			node = node.parent


def optimal_sample(alg, num_chosen, difficulty = 'optimal'):
	optimal_cost = alg.optimal_plan[-1].g
	if difficulty == 'optimal':
		optimal_plan = alg.optimal_plan.copy()
	else:
		if difficulty == 'hard':
			optimal_plan = alg.optimal_plan[:len(alg.optimal_plan)//3]
		elif difficulty == 'med':
			optimal_plan = alg.optimal_plan[len(alg.optimal_plan)//3: 2 * len(alg.optimal_plan)//3]
		elif difficulty == 'easy':
			optimal_plan = alg.optimal_plan[2*len(alg.optimal_plan)//3:]
		elif difficulty == 'easy_med':
			optimal_plan = alg.optimal_plan[len(alg.optimal_plan)//3:]
	nodes = random.sample(optimal_plan, min(num_chosen, len(optimal_plan)))
	optimal_costs = [optimal_cost - node.g for node in nodes]
	return nodes, optimal_costs

def create_supervision(params, solver = None):
	def get_puzzle_str(alg, node):
		if params.domain == 'sokoban':
			puzzle_str = convert_array_to_sb(alg.puzzle, alg.docks, node.boxes, node.pos)
		elif params.domain == 'maze':
			puzzle_str = convert_array_to_maze(alg.puzzle, node.pos, alg.goal)
		return puzzle_str

	solver = AStar_sokoban if params.domain == 'sokoban' else AStar_maze if solver is None else solver
	path = f'{params.data_dir}/{params.dataset}'	
	for split in ['train', 'val']:
		seqs_per_puzzle = params.train_seqs if split == 'train' else params.val_seqs
		if len(params.alg_files) > 0:
			alg_files = [f'{path}/{split}/{alg_file}.pkl' for alg_file in params.alg_files]
		alg_files = glob.glob(f'{path}/{split}/*.pkl')
		for alg_file in alg_files:
			if params.domain == 'sokoban':
				try:
					incorrect_file = str(params.create_data[3]) not in alg_file or 'supervised' in alg_file
				except:
					incorrect_file = str(params.bootstrap_data[3]) not in alg_file or 'supervised' in alg_file
			else:
				incorrect_file = 'supervised' in alg_file

			if incorrect_file:
				continue
			dataset = []
			with open(alg_file, 'rb') as f:
				algs = pkl.load(f)
			for alg in tqdm(algs, desc = alg_file):
				initial_str = get_puzzle_str(alg, alg.closed[0])
				closed_set = alg.closed.copy()
				num_chosen, iters = seqs_per_puzzle, 0
				while num_chosen > 0 and len(closed_set) > 0:
					nodes, optimal_costs = optimal_sample(alg, num_chosen, difficulty = params.sample.split('optimal_')[-1])
					num_chosen -= len(nodes)
					for node, optimal_cost in zip(nodes, optimal_costs):
						dataset.append((initial_str, get_puzzle_str(alg, node), node.h, optimal_cost))
			alg_file = alg_file.split('/')[-1].replace('.pkl', f'_{params.sample}.pkl')
			with open(f'{path}/{split}/supervised_{alg_file}', 'wb') as f:
				pkl.dump(dataset, f)
			print(f'Created {len(dataset)}')

def tokenize_data(params, datapoints, tokenizer, filename):
	legend = {'maze': "@ - player, # - wall, . - empty cell, X - goal", 'sokoban': "@ - player, # - wall, . - empty docks, ' ' - empty cell, $ - box, X - box on dock, O - player on dock"}
	data_inputs, data_labels, data_decoder_inputs = [], [], []
	with open(params.prompt_file) as f:
		prompt = f.read()
	for datapoint in tqdm(datapoints, desc = filename):
		initial_str, puzzle_str, heuristic, optimal_cost = datapoint
		pred_diff = None
		if isinstance(heuristic, tuple):
			heuristic, pred_diff = heuristic
		difference = optimal_cost - heuristic
		input_prompt = prompt.replace('{puzzle_str}', puzzle_str).replace('{heuristic}', str(int(heuristic))).replace('{initial_str}', initial_str)
		input_prompt = input_prompt.replace('{puzzle_legend}', legend[params.domain]).replace('{domain}', params.domain)
		input_ids = tokenizer(input_prompt).input_ids
		label_str = str(int(difference)) if isinstance(difference, int) else str(round(difference, 2))
		if pred_diff is not None:
			label_str = f'{int(pred_diff)}, {optimal_cost - heuristic - pred_diff}'
		labels = tokenizer(label_str, add_special_tokens = 't5' in params.base_model).input_ids
		decoder_input_ids = None
		# if error is None:
		# 	labels = tokenizer(label_str, add_special_tokens = 't5' in params.base_model).input_ids
		# 	decoder_input_ids = None
		# else:
		# 	prefix_label_str = label_str
		# 	label_str = f', error = {optimal_cost - heuristic - error}'
		# 	prefix_label_tokens = tokenizer(prefix_label_str, add_special_tokens = False).input_ids
		# 	label_tokens = tokenizer(label_str, add_special_tokens = False).input_ids
		# 	decoder_input_ids = [tokenizer.bos_token_id] + prefix_label_tokens + label_tokens
		# 	labels = [-100] * len(prefix_label_tokens) + label_tokens + [tokenizer.eos_token_id] # shifted right
		data_inputs.append(input_ids)
		data_labels.append(labels)
		if decoder_input_ids is not None:
			data_decoder_inputs.append(decoder_input_ids)
	with open(filename, 'wb') as f:
		if len(data_decoder_inputs) > 0:
			pkl.dump((data_inputs, data_labels, data_decoder_inputs), f)
			return data_inputs, data_labels, data_decoder_inputs
		else:
			pkl.dump((data_inputs, data_labels), f)
			return data_inputs, data_labels

def read_data(params, tokenizer):
	print('Loading data...')
	data = {'train': {'raw': [], 'tokenized': ([], [], [])}, 'val': {'raw': [], 'tokenized': ([], [], [])}, 'test': {'raw': [], 'alg': []}}
	path = f'{params.data_dir}/{params.dataset}'
	for split in ['train', 'val']:
		files = params.val_files if split == 'val' else params.train_files
		for file in files:
			with open(f'{path}/{split}/supervised_{file}.pkl', 'rb') as f:
				datapoints = pkl.load(f)
				data[split]['raw'].extend(datapoints)
			
			tokenized_file = f"{params.data_dir}/{params.dataset}/{split}/{params.base_model}/tokenized_{file}_{params.prompt_file.replace('.txt', '')}.pkl"
			if os.path.exists(tokenized_file):
				with open(tokenized_file, 'rb') as f:
					tokenized_data = pkl.load(f)
			else:
				os.makedirs(f"{params.data_dir}/{params.dataset}/{split}/{params.base_model}", exist_ok = True)
				tokenized_data = tokenize_data(params, datapoints, tokenizer, tokenized_file)
			for i in range(len(tokenized_data)):
				data[split]['tokenized'][i].extend(tokenized_data[i])
		if len(data[split]['tokenized'][-1]) == 0:
			data[split]['tokenized'] = data[split]['tokenized'][:-1]
	
	ilr_split = params.test_ilr[0]
	for file in params.test_ilr[1:]:
		with open(f'{path}/{ilr_split}/{file}.txt') as f:
			data['test']['raw'].extend(f.read().split(';'))
		
		with open(f"{path}/{ilr_split}/alg_{file}.pkl", 'rb') as f:
			data['test']['alg'].extend(pkl.load(f))
	return data