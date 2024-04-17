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
from ddp import *

def generate_maze(size):
	maze = np.zeros((size, size))
	wall_percent = np.random.uniform(0.3, 0.5)

	# This guarantees 30 - 50% walls
	wall_number = int(wall_percent * size * size)
	wall_x = np.random.choice(np.arange(size), wall_number)
	wall_y = np.random.choice(np.arange(size), wall_number)
	maze[wall_x, wall_y] = 1

	open_pos = (1 - maze).nonzero()
	start, goal = np.random.choice(np.arange(len(open_pos[0])), 2, replace = False)
	start = open_pos[0][start], open_pos[1][start]
	goal = open_pos[0][goal], open_pos[1][goal]
	return convert_array_to_maze(maze, start, goal)

def create_maze_dataset(params, num_train, num_val, num_test, size, solver = None):
	solver = AStar_maze if solver is None else solver
	puzzles = []
	p_bar = tqdm(range(num_train + num_val))
	mazes = set()
	while len(puzzles) < num_train + num_val:
		maze = generate_maze(size)
		alg = solver(maze)
		if maze not in mazes:
			mazes.add(maze)
			alg.search()
			if alg.optimal_plan is not None:
				if len(alg.optimal_plan) > size and len(alg.optimal_plan)/alg.closed[0].h > 1.3:
					puzzles.append((maze, alg))
					p_bar.n += 1
					p_bar.refresh()
	
	random.shuffle(puzzles)
	train, val, test = puzzles[:num_train], puzzles[num_train:], []
	p_bar = tqdm(range(num_test))
	while len(test) < num_test:
		maze = generate_maze(size)
		alg = AStar_maze(maze)
		if maze not in mazes:
			mazes.add(maze)
			alg.search()
			if alg.optimal_plan is not None:
				if len(alg.optimal_plan) > size and len(alg.optimal_plan)/alg.closed[0].h > 1.05:
					test.append((maze, alg))
					p_bar.n += 1
					p_bar.refresh()

	print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
	os.makedirs(f'{params.data_dir}/{params.dataset}/train', exist_ok = True)
	os.makedirs(f'{params.data_dir}/{params.dataset}/val', exist_ok = True)
	os.makedirs(f'{params.data_dir}/{params.dataset}/test', exist_ok = True)
	with open(f'{params.data_dir}/{params.dataset}/train/mazes_{size}.txt', 'w') as f:
		f.write(';'.join([x[0] for x in train]))
	
	with open(f'{params.data_dir}/{params.dataset}/val/mazes_{size}.txt', 'w') as f:
		f.write(';'.join([x[0] for x in val]))
	
	with open(f'{params.data_dir}/{params.dataset}/test/mazes_{size}.txt', 'w') as f:
		f.write(';'.join([x[0] for x in test]))

	with open(f'{params.data_dir}/{params.dataset}/train/alg_mazes_{size}.pkl', 'wb') as f:
		pkl.dump([x[1] for x in train], f)
	
	with open(f'{params.data_dir}/{params.dataset}/val/alg_mazes_{size}.pkl', 'wb') as f:
		pkl.dump([x[1] for x in val], f)
	
	with open(f'{params.data_dir}/{params.dataset}/test/alg_mazes_{size}.pkl', 'wb') as f:
		pkl.dump([x[1] for x in test], f)

# def create_maze_dataset(params, num_train, num_val, num_test, size):
# 	puzzles = []
# 	p_bar = tqdm(range(num_train + num_val + num_test))
# 	mazes = set()
# 	while len(puzzles) < num_train + num_val + num_test:
# 		maze = generate_maze(size)
# 		alg = AStar_maze(maze)
# 		if maze not in mazes:
# 			mazes.add(maze)
# 			alg.search()
# 			if alg.optimal_plan is not None:
# 				if len(alg.optimal_plan) > size and len(alg.optimal_plan)/alg.closed[0].h > 1.3:
# 					puzzles.append((maze, alg))
# 					p_bar.n += 1
# 					p_bar.refresh()
	
# 	random.shuffle(puzzles)
# 	train, val, test = puzzles[:num_train], puzzles[num_train: num_train + num_val], puzzles[num_train + num_val:]
# 	print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
# 	os.makedirs(f'{params.data_dir}/{params.dataset}/train', exist_ok = True)
# 	os.makedirs(f'{params.data_dir}/{params.dataset}/val', exist_ok = True)
# 	os.makedirs(f'{params.data_dir}/{params.dataset}/test', exist_ok = True)
# 	with open(f'{params.data_dir}/{params.dataset}/train/mazes_{size}.txt', 'w') as f:
# 		f.write(';'.join([x[0] for x in train]))
	
# 	with open(f'{params.data_dir}/{params.dataset}/val/mazes_{size}.txt', 'w') as f:
# 		f.write(';'.join([x[0] for x in val]))
	
# 	with open(f'{params.data_dir}/{params.dataset}/test/mazes_{size}.txt', 'w') as f:
# 		f.write(';'.join([x[0] for x in test]))
	

# 	with open(f'{params.data_dir}/{params.dataset}/train/alg_mazes_{size}.pkl', 'wb') as f:
# 		pkl.dump([x[1] for x in train], f)
	
# 	with open(f'{params.data_dir}/{params.dataset}/val/alg_mazes_{size}.pkl', 'wb') as f:
# 		pkl.dump([x[1] for x in val], f)
	
# 	with open(f'{params.data_dir}/{params.dataset}/test/alg_mazes_{size}.pkl', 'wb') as f:
# 		pkl.dump([x[1] for x in test], f)

def read_boxoban(params, split, hardness = 'unfiltered'):
	path = f'{params.data_dir}/boxoban-levels/{hardness}/{split}/'
	files = os.listdir(path)
	puzzles = []
	for file in tqdm(files, desc = split):
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
						annotate_deceptive_nodes(alg)
						algs.append(alg)
						p_bar.n += 1
						p_bar.refresh()
			print(f'{split}: {len(astar)}')
			path = f'{params.data_dir}/{params.dataset}/{split}/'
			os.makedirs(path, exist_ok = True)
			with open(path + f'sokoban_{subsample}.txt', 'w') as f:
				f.write(';'.join(astar))

			with open(path + f'alg_sokoban_{subsample}.pkl', 'wb') as f:
				pkl.dump(algs, f)

# def create_supervision(params, solver = None):
# 	solver = AStar_sokoban if params.domain == 'sokoban' else AStar_maze if solver is None else solver
# 	path = f'{params.data_dir}/{params.dataset}'	
# 	for split in ['train', 'val']:
# 		alg_files = glob.glob(f'{path}/{split}/*.pkl')
# 		for alg_file in alg_files:
# 			try:
# 				incorrect_file = str(params.create_data[3]) not in alg_file or 'supervised' in alg_file
# 			except:
# 				incorrect_file = str(params.bootstrap_data[3]) not in alg_file or 'supervised' in alg_file
# 			if incorrect_file:
# 				continue
# 			dataset = []
# 			with open(alg_file, 'rb') as f:
# 				algs = pkl.load(f)
# 			for alg in tqdm(algs, desc = alg_file):
# 				closed_set = alg.closed.copy()
# 				random.shuffle(closed_set)
# 				num_chosen = 5 if split == 'val' else 15
# 				optimal_plan = alg.optimal_plan.copy()
# 				nodes = random.sample(optimal_plan, min(num_chosen, len(optimal_plan)))
# 				indices = [optimal_plan.index(node) for node in nodes]
# 				optimal_costs = [len(optimal_plan) - idx - 1 for idx in indices]
# 				for node, optimal_cost in zip(nodes, optimal_costs):
# 					puzzle_str = convert_array_to_sb(alg.puzzle, alg.docks, node.boxes, node.pos)
# 					dataset.append((puzzle_str, node.h, optimal_cost))
# 			alg_file = alg_file.split('/')[-1]
# 			with open(f'{path}/{split}/supervised_{alg_file}', 'wb') as f:
# 				pkl.dump(dataset, f)
# 			print(f'Created {len(dataset)}')

def annotate_deceptive_nodes(alg):
	counted_nodes = set()
	for frontier in alg.frontier_snapshots:
		# Getting deceptive nodes for one iteration
		deceptive_nodes = set()
		for node in frontier:
			if node.is_optimal:
				break
			else:
				deceptive_nodes.add(node)
		
		# Incrementing the deception_score of unaccounted deceptive nodes, 
		# and their parents until we reach an optimal node
		# Don't go up more than 3 generations(blame your problems on your great-grandparents)
		for node in deceptive_nodes - counted_nodes:
			num_generations = 3
			while not node.is_optimal and num_generations > 0:
				node.deception_score += 1
				node = node.parent
				num_generations -= 1
		counted_nodes = counted_nodes.union(deceptive_nodes)

def create_supervision(params, solver = None):
	
	def random_sample(closed_set):
		return closed_set.pop(0)
	
	def deception_sample(closed_set):
		scores = torch.tensor([node.deception_score for node in closed_set], dtype = torch.float64)
		weights = torch.softmax(scores, -1)
		node = np.random.choice(closed_set, p = weights)
		closed_set.remove(node)
		return node
	
	def optimal_sample(alg, num_chosen):
		optimal_plan = alg.optimal_plan.copy()
		nodes = random.sample(optimal_plan, min(num_chosen, len(optimal_plan)))
		indices = [optimal_plan.index(node) for node in nodes]
		optimal_costs = [len(optimal_plan) - idx for idx in indices]
		# datapoints = []
		# for node, optimal_cost in zip(nodes, optimal_costs):
		# 	datapoints.append((puzzle_str, node.h, optimal_cost))
		return nodes, optimal_costs
	
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
		alg_files = glob.glob(f'{path}/{split}/*.pkl')
		for alg_file in alg_files:
			try:
				incorrect_file = str(params.create_data[3]) not in alg_file or 'supervised' in alg_file
			except:
				incorrect_file = str(params.bootstrap_data[3]) not in alg_file or 'supervised' in alg_file
			if incorrect_file:
				continue
			dataset = []
			with open(alg_file, 'rb') as f:
				algs = pkl.load(f)
			for alg in tqdm(algs, desc = alg_file):
				initial_str = get_puzzle_str(alg, alg.closed[0])
				closed_set = alg.closed.copy()
				if 'rand' in params.sample:
					random.shuffle(closed_set)
				annotate_deceptive_nodes(alg)
				num_chosen, iters = seqs_per_puzzle, 0
				while num_chosen > 0 and len(closed_set) > 0:
					if params.sample == 'random':
						node = random_sample(closed_set)

					elif params.sample == 'optimal':
						nodes, optimal_costs = optimal_sample(alg, num_chosen)
						num_chosen -= len(nodes)
						for node, optimal_cost in zip(nodes, optimal_costs):
							dataset.append((initial_str, get_puzzle_str(alg, node), node.h, optimal_cost, node.deception_score))
						node = nodes[0]

					elif params.sample == 'deception':
						node = deception_sample(closed_set)

					elif 'rand10_' in params.sample:
						if num_chosen > (seqs_per_puzzle - 10): # seqs_per_puzzle // 2:
							node = random_sample(closed_set)
						else:
							if 'opt' in params.sample: # rand_opt
								nodes, optimal_costs = optimal_sample(alg, num_chosen)
								num_chosen -= len(nodes)
								for node, optimal_cost in zip(nodes, optimal_costs):
									dataset.append((initial_str, get_puzzle_str(alg, node), node.h, optimal_cost, node.deception_score))
							elif 'dec' in params.sample: # rand_dec
								node = deception_sample(closed_set)
							
					elif params.sample == 'opt_dec10':
						if num_chosen > (seqs_per_puzzle - 10): # seqs_per_puzzle // 2:
							node = deception_sample(closed_set)
						else:
							nodes, optimal_costs = optimal_sample(alg, num_chosen)
							num_chosen -= len(nodes)
							for node, optimal_cost in zip(nodes, optimal_costs):
								dataset.append((initial_str, get_puzzle_str(alg, node), node.h, optimal_cost, node.deception_score))

					if node not in alg.optimal_plan:
						puzzle_str = get_puzzle_str(alg, node)
						new_alg = solver(puzzle_str, terminate_after = 7000)
						new_alg.search()
						if new_alg.optimal_plan is not None:
							optimal_cost = len(new_alg.optimal_plan)
							dataset.append((initial_str, puzzle_str, node.h, optimal_cost, node.deception_score))
							num_chosen -= 1
							iters = 0
					iters += 1
			alg_file = alg_file.split('/')[-1].replace('.pkl', f'_{params.sample}{params.target}.pkl')
			with open(f'{path}/{split}/supervised_{alg_file}', 'wb') as f:
				pkl.dump(dataset, f)
			print(f'Created {len(dataset)}')


def tokenize_data(params, datapoints, tokenizer, filename):
	legend = {'maze': "@ - player, # - wall, . - empty cell, X - goal", 'sokoban': "@ - player, # - wall, . - empty docks, ' ' - empty cell, $ - box, X - box on dock, O - player on dock"}
	data_inputs, data_labels = [], []
	with open(params.prompt_file) as f:
		prompt = f.read()
	for datapoint in tqdm(datapoints, desc = filename):
		if len(datapoint) == 3:
			puzzle_str, heuristic, optimal_cost = datapoint
			initial_str, deception_score = '', 0
		elif len(datapoint) == 5:
			initial_str, puzzle_str, heuristic, optimal_cost, deception_score = datapoint
			if params.target == '':
				initial_str, deception_score = '', 0
		# optimal_cost is actually counting one extra step, since start and goal are in the plan for maze. In sokoban, you don't end at the goal
		difference = optimal_cost + deception_score - heuristic
		input_prompt = prompt.replace('{puzzle_str}', puzzle_str).replace('{heuristic}', str(int(heuristic))).replace('{initial_str}', initial_str)
		input_prompt = input_prompt.replace('{puzzle_legend}', legend[params.domain]).replace('{domain}', params.domain)
		input_ids = tokenizer(input_prompt).input_ids
		labels = tokenizer(str(int(difference)), add_special_tokens = 't5' in params.base_model).input_ids
		data_inputs.append(input_ids)
		data_labels.append(labels)
	with open(filename, 'wb') as f:
		pkl.dump((data_inputs, data_labels), f)
	return data_inputs, data_labels

def read_data(params, tokenizer):
	if is_main_process():
		print('Loading data...')
	data = {'train': {'raw': [], 'tokenized': ([], [])}, 'val': {'raw': [], 'tokenized': ([], [])}, 'test': {'raw': [], 'alg': []}}
	path = f'{params.data_dir}/{params.dataset}'
	for split in ['train', 'val']:
		for file in params.train_files: # ['alg_mazes_5.pkl', 'alg_mazes_7.pkl', 'alg_mazes_10.pkl']
			with open(f'{path}/{split}/supervised_{file}.pkl', 'rb') as f:
				datapoints = pkl.load(f)
				data[split]['raw'].extend(datapoints)
			
			tokenized_file = f"{params.data_dir}/{params.dataset}/{split}/{params.base_model}/tokenized_{file}.pkl"
			if os.path.exists(tokenized_file):
				with open(tokenized_file, 'rb') as f:
					input_ids, labels = pkl.load(f)
			else:
				os.makedirs(f"{params.data_dir}/{params.dataset}/{split}/{params.base_model}", exist_ok = True)
				input_ids, labels = tokenize_data(params, datapoints, tokenizer, tokenized_file)
			data[split]['tokenized'][0].extend(input_ids)
			data[split]['tokenized'][1].extend(labels)
	
	ilr_split = params.test_ilr[0]
	for file in params.test_ilr[1:]: # ['mazes_5', 'mazes_7', 'mazes_10']:
		with open(f'{path}/{ilr_split}/{file}.txt') as f:
			data['test']['raw'].extend(f.read().split(';'))
		
		with open(f"{path}/{ilr_split}/alg_{file}.pkl", 'rb') as f:
			data['test']['alg'].extend(pkl.load(f))
	return data