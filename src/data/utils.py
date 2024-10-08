from tqdm import tqdm
import random
import os
import pickle as pkl
import numpy as np
from algorithm.utils import *
from algorithm import AStar_maze, AStar_sokoban, AStar_stp, Dijkstra_maze, Node_stp
import glob
from transformers import AutoTokenizer
import torch
import sys
sys.path.append('data/mazelib')
from mazelib import Maze
from mazelib.generate.Prims import Prims
import warnings
sys.path.append('../')
from model import T5ImprovedHeuristic
from .dataset import T5HeuristicDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
model_dict = {'phi2': 'microsoft/phi-2', 'codellama': 'codellama/CodeLlama-7b-hf', 'flan-t5': 'google/flan-t5-small', 'code-t5': 'Salesforce/codet5-small', 't5': 'google-t5/t5-small', 'code-t5-base': 'Salesforce/codet5-base', 'code-t5-large': 'Salesforce/codet5-large'}


def generate_maze(width, height):
	m = Maze()
	m.generator = Prims(width, height)
	m.generate()
	return m

def generate_multipath_maze(maze):
	d_alg = Dijkstra_maze(maze)
	dist_s = d_alg.fill(from_start=True)
	dist_g = d_alg.fill(from_start=False)
	grouped = np.where(dist_s <= dist_g, 1, 2) # starts are 1, goals are 2
	grouped[d_alg.puzzle == 1] = 0
	patterns = [[1, 0, 2]]
	wall_breaking_thresolds = [0.3]
	walls_broken = 0
	for pattern, wall_breaking_threshold in zip(patterns, wall_breaking_thresolds):
		for row in range(d_alg.size[0]): # Iterating over each row
			for i in range(len(grouped[row]) - len(pattern)): # Iterating over each subarray in a row
				if np.array_equal(grouped[row, i:i + len(pattern)], pattern) or np.array_equal(grouped[row, i:i + len(pattern)], pattern[::-1]):
					for j in range(len(pattern)):  # If pattern is found, breaking the walls in puzzle where pattern_indices = 0
						if pattern[j] == 0:
							assert d_alg.puzzle[row, i + j] == 1
							if random.random() > wall_breaking_threshold:
								d_alg.puzzle[row, i + j] = 0
								walls_broken += 1

		for col in range(d_alg.size[1]):
			for i in range(len(grouped[col]) - len(pattern)):
				if np.array_equal(grouped.T[col, i:i + len(pattern)], pattern) or np.array_equal(grouped.T[col, i:i + len(pattern)], pattern[::-1]):
					for j in range(len(pattern)):
						if pattern[j] == 0:
							assert d_alg.puzzle[i + j, col] == 1
							if random.random() > wall_breaking_threshold:
								d_alg.puzzle[i + j, col] = 0
								walls_broken += 1
	return convert_array_to_maze(d_alg.puzzle, d_alg.start, d_alg.goal), walls_broken

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
			maze, walls_broken = generate_multipath_maze(maze)
			alg = AStar_maze(maze)
			if maze not in mazes:
				mazes.add(maze)
				alg.search()
				if alg.optimal_plan is not None:
					if len(alg.optimal_plan) > 2*size and len(alg.closed)/len(alg.optimal_plan) > 3.5:
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

def create_sokoban_dataset(params, num_train, num_val, num_test, subsample, terminate_after = 7000, min_iterations = 0, optimal_length = 20, solver = None):
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
					if alg.iterations > min_iterations and len(alg.optimal_plan) > optimal_length and len(alg.closed)/len(alg.optimal_plan) > 6:
						puzzle = convert_array_to_sb(alg.puzzle, alg.docks, alg.initial_boxes, alg.initial_pos)
						astar.append(puzzle)
						algs.append(alg)
						p_bar.n += 1
						p_bar.refresh()
			print(f'{split}: {len(astar)}')
			path = f'{params.data_dir}/{params.dataset}/{split}/'
			os.makedirs(path, exist_ok = True)

			file_suffix = f'{subsample}_{optimal_length}_{min_iterations}'
			with open(path + f'sokoban_{file_suffix}.txt', 'w') as f:
				f.write(';'.join(astar))

			with open(path + f'alg_sokoban_{file_suffix}.pkl', 'wb') as f:
				pkl.dump(algs, f)

def is_stp_solvable(puzzle, size):    
    # Count inversions
    inversions = 0
    for i in range(len(puzzle)):
        for j in range(i + 1, len(puzzle)):
            if puzzle[i] != 0 and puzzle[j] != 0 and puzzle[i] > puzzle[j]:
                inversions += 1
    
    # Find the row of the blank tile (0) from the bottom
    blank_row = size - (puzzle.index(0) // size)
    
    # Check solvability based on puzzle size
    if size % 2 == 1:
        # Odd-sized puzzle is solvable if number of inversions is even
        return inversions % 2 == 0
    else:
        # Even-sized puzzle is solvable if:
        # - the blank is on an even row counting from the bottom and number of inversions is odd
        # - or the blank is on an odd row counting from the bottom and number of inversions is even
        return (blank_row % 2 == 0 and inversions % 2 == 1) or \
               (blank_row % 2 == 1 and inversions % 2 == 0)


def generate_stp(width):
	# Ref: https://github.com/pyGuru123/Python-Games/blob/a3817dd31055d9208a3f9899ff1c2c5cfb9a33e8/Picture%20Sliding%20Puzzle/game.py#L81
	puzzle = [i for i in range(1,width * width)] + [0]
	random.shuffle(puzzle)
	while not is_stp_solvable(puzzle, width):
		random.shuffle(puzzle)
	puzzle = map(lambda x : str(x), puzzle)
	return ' '.join(puzzle) + '\n'

def generate_stp_simulate(width):
	goal_number = iter(range(width * width))
	puzzle = np.zeros((width, width), dtype = np.int32)
	for i in range(width):
		for j in range(width):
			g = next(goal_number)
			puzzle[i, j] = g
	num_steps = random.randint(20, 30)
	directions = (1, 0), (-1, 0), (0, 1), (0, -1)
	empty_pos, last_pos = (0, 0), (-1, -1)
	for _ in range(num_steps):
		next_pos = []
		for next_x, next_y in directions:
			possible_pos = (empty_pos[0] + next_x, empty_pos[1] + next_y)
			if 0 <= possible_pos[0] < width and 0 <= possible_pos[1] < width and possible_pos != last_pos:
				next_pos.append(possible_pos)
		next_pos = random.choice(next_pos)
		puzzle[next_pos], puzzle[empty_pos] = puzzle[empty_pos], puzzle[next_pos]
		last_pos, empty_pos = empty_pos, next_pos
	puzzle = puzzle.flatten().tolist()
	# breakpoint()
	# assert is_stp_solvable(puzzle, width)
	puzzle = map(lambda x : str(x), puzzle)
	return ' '.join(puzzle) + '\n'

def create_stp_dataset(params, num_train, num_val, num_test, width, terminate_after = 5000, min_iterations = 0, optimal_length = 20, solver = None):
	solver = AStar_stp if solver is None else solver
	p_bar = tqdm(range(num_train + num_val + num_test))
	stps = set()
	astar, algs = [], []
	while len(astar) < num_train + num_val + num_test:
		stp = generate_stp(width) if width == 3 else generate_stp_simulate(width)
		if stp not in stps:
			stps.add(stp)
			stp = convert_stp_to_chars(stp)
			alg = solver(stp, terminate_after = terminate_after)
			alg.search()
			if alg.optimal_plan is not None:
				if alg.iterations > min_iterations and len(alg.optimal_plan) > optimal_length and len(alg.closed)/len(alg.optimal_plan) > 6:
					astar.append(stp)
					algs.append(alg)
					p_bar.n += 1
					p_bar.refresh()

	for split, num_puzzles in zip(['train', 'val', 'test'], [num_train, num_val, num_test]):
		if num_puzzles > 0:
			path = f'{params.data_dir}/{params.dataset}/{split}/'
			os.makedirs(path, exist_ok = True)
			
			puzzles = astar[:num_puzzles]
			astar = astar[num_puzzles:]
			with open(path + f'stp_{width}.txt', 'w') as f:
				f.write(';'.join(puzzles))

			puzzle_algs = algs[:num_puzzles]
			algs = algs[num_puzzles:]
			with open(path + f'alg_stp_{width}.pkl', 'wb') as f:
				pkl.dump(puzzle_algs, f)

def optimal_sample(alg, num_chosen, params, split, difficulty = 'optimal'):
	optimal_cost = alg.optimal_plan[-1].g
	if difficulty == 'optimal' or difficulty == 'dist':
		optimal_plan = alg.optimal_plan.copy()
	else:
		optimal_hard = alg.optimal_plan[:len(alg.optimal_plan)//3]
		optimal_med = alg.optimal_plan[len(alg.optimal_plan)//3: 2 * len(alg.optimal_plan)//3]
		optimal_easy = alg.optimal_plan[2 * len(alg.optimal_plan)//3:]
		split_choice = random.random() if split == 'train' else 0
		if difficulty == 'hard':
			optimal_plan = optimal_hard if split_choice < 0.99 else optimal_easy + optimal_med
		elif difficulty == 'med':
			optimal_plan = optimal_med if split_choice < 0.99 else optimal_easy + optimal_hard
		elif difficulty == 'easy':
			optimal_plan = optimal_easy if split_choice < 0.99 else optimal_med + optimal_hard
		elif difficulty == 'easy_med':
			optimal_plan = optimal_easy + optimal_med if split_choice < 0.99 else optimal_hard
		elif difficulty == 'easy_hard':
			optimal_plan = optimal_easy + optimal_hard if split_choice < 0.99 else optimal_med
		elif difficulty == 'med_hard':
			optimal_plan = optimal_med + optimal_hard if split_choice < 0.99 else optimal_easy
	if difficulty == 'dist':
		dist_factor = params.dist_factor
		w = (1/dist_factor) * torch.log(torch.tensor([len(optimal_plan)/i for i in range(len(optimal_plan), 0, -1)])) # Log
		# w = (1/dist_factor) * torch.tensor([len(optimal_plan)/i for i in range(len(optimal_plan), 0, -1)]) # No Log
		# w = (1/dist_factor) * torch.tensor([i/len(optimal_plan) for i in range(1, len(optimal_plan) + 1)]) # Lin
		w = torch.softmax(w, dim = 0).numpy()
		nodes = np.random.choice(optimal_plan, replace = False, size = min(num_chosen, len(optimal_plan)), p = w)
	else:
		nodes = random.sample(optimal_plan, min(num_chosen, len(optimal_plan)))
	optimal_costs = [optimal_cost - node.g for node in nodes]
	return nodes, optimal_costs

def get_puzzle_str(domain, alg, node):
	if domain == 'sokoban':
		puzzle_str = convert_array_to_sb(alg.puzzle, alg.docks, node.boxes, node.pos)
	elif domain == 'maze':
		puzzle_str = convert_array_to_maze(alg.puzzle, node.pos, alg.goal)
	elif domain == 'stp':
		puzzle_str = convert_array_to_stp(node.pos)
	return puzzle_str

@torch.no_grad()
def semdedup(alg, num_chosen, params, model):
	import time
	optimal_cost = alg.optimal_plan[-1].g
	nodes = alg.optimal_plan.copy()
	node_details = [(get_puzzle_str(params.domain, alg, alg.closed[0]), get_puzzle_str(params.domain, alg, node), node.h, optimal_cost - node.g) for node in nodes]
	params.loss = 'l2'
	tokenized_data = tokenize_data(params, node_details, model.tokenizer, 'semdedup', disable_bar = True)
	node_dataset = T5HeuristicDataset(params, tokenized_data, nodes, model.tokenizer)
	node_dl = DataLoader(node_dataset, batch_size = 64, num_workers = 0, pin_memory = False, shuffle = False, collate_fn = node_dataset.collate_fn_train)
	assert len(nodes) <= 64, "multi-batch needed"
	embs = model.get_embeddings(node_dl)
	centroid = embs.mean(dim = 0).unsqueeze(0)
	dist_to_centroid = F.cosine_similarity(embs, centroid)
	embs = embs[torch.argsort(dist_to_centroid, descending = True)]
	embs = F.normalize(embs)
	pairwise_sims = embs @ embs.T
	triu_sim = torch.triu(pairwise_sims, diagonal = 1)
	M = torch.max(triu_sim, dim = 0)[0]
	w = (1/params.dist_factor) * torch.log(torch.tensor([len(alg.optimal_plan)/i for i in range(len(alg.optimal_plan), 0, -1)]))
	w = torch.softmax(w, dim = 0).numpy()
	_, inds1 = torch.topk(M, largest = False, k = num_chosen)
	inds1 = inds1.cpu().numpy()
	inds2 = np.random.choice(range(len(alg.optimal_plan)), replace = False, size = min(num_chosen, len(alg.optimal_plan)), p = w)
	inds, counts = np.unique(np.concatenate((inds1, inds2)), return_counts = True)
	inds = np.random.choice(inds, size = num_chosen, p = counts/counts.sum(), replace = False)
	nodes = [nodes[i] for i in inds]
	optimal_costs = [optimal_cost - node.g for node in nodes]
	return nodes, optimal_costs

def create_supervision(params, solver = None):
	if solver is None:
		if params.domain == 'sokoban':
			solver = AStar_sokoban
		elif params.domain == 'maze':
			solver = AStar_maze
		elif params.domain == 'stp':
			solver = AStar_stp

	path = f'{params.data_dir}/{params.dataset}'
	if 'semdedup' in params.sample:
		model = T5ImprovedHeuristic(params, model_dict[params.base_model], params.device)
		model = model.to(torch.device(params.device))
	for split in ['train', 'val']:
		seqs_per_puzzle = params.train_seqs if split == 'train' else params.val_seqs
		alg_files = [f'{path}/{split}/{alg_file}.pkl' for alg_file in params.alg_files] if len(params.alg_files) else glob.glob(f'{path}/{split}/*.pkl')
		for alg_file in alg_files:
			incorrect_file = str(params.create_data[3]) not in alg_file or 'supervised' in alg_file if params.domain == 'sokoban' else 'supervised' in alg_file
			if not incorrect_file:
				dataset = []
				try:
					with open(alg_file, 'rb') as f:
						algs = pkl.load(f)
				except Exception as e:
					print(str(e))
					continue
				for i, alg in enumerate(tqdm(algs, desc = alg_file)):
					initial_str = get_puzzle_str(params.domain, alg, alg.closed[0])
					closed_set = alg.closed.copy()
					if 'optimal' in params.sample:
						nodes, optimal_costs = optimal_sample(alg, seqs_per_puzzle, params, split, difficulty = params.sample.split('optimal_')[-1])
					elif 'semdedup' in params.sample:
						nodes, optimal_costs = semdedup(alg, seqs_per_puzzle, params, model)
					for node, optimal_cost in zip(nodes, optimal_costs):
						dataset.append((initial_str, get_puzzle_str(params.domain, alg, node), node.h, optimal_cost))
				if params.sample == 'optimal_dist':
					sample = params.sample + '_' + str(params.dist_factor)
				else:
					sample = params.sample
				alg_file = alg_file.split('/')[-1].replace('.pkl', f'_{sample}_{seqs_per_puzzle}.pkl')
				with open(f'{path}/{split}/supervised_{alg_file}', 'wb') as f:
					pkl.dump(dataset, f)
				print(f'Created {len(dataset)}')

def tokenize_data(params, datapoints, tokenizer, filename, disable_bar = False):
	legend = {'maze': "@ - player, # - wall, . - empty cell, X - goal", 'sokoban': "@ - player, # - wall, . - empty docks, ' ' - empty cell, $ - box, X - box on dock, O - player on dock", 'stp': '0 - empty space'}
	data_inputs, data_labels, data_decoder_inputs = [], [], []
	with open(params.prompt_file) as f:
		prompt = f.read()
	for datapoint in tqdm(datapoints, desc = filename, disable = disable_bar):
		initial_str, puzzle_str, heuristic, optimal_cost = datapoint
		if params.domain == 'stp':
			stp_goal = get_stp_goal(puzzle_str)
			puzzle_str = puzzle_str.strip() + '"\ngoal = "' + stp_goal.strip()
		difference = optimal_cost - heuristic
		input_prompt = prompt.replace('{puzzle_str}', puzzle_str).replace('{heuristic}', str(int(heuristic))).replace('{initial_str}', initial_str)
		input_prompt = input_prompt.replace('{puzzle_legend}', legend[params.domain]).replace('{domain}', params.domain)
		input_ids = tokenizer(input_prompt).input_ids
		label_str = f'{int(difference)}'
		
		if params.loss == 'lm':
			labels = tokenizer(label_str, add_special_tokens = 't5' in params.base_model).input_ids
			decoder_input_ids = labels[:-1]
			labels = labels[1:]

		if params.loss == 'l2':
			if 't5' not in params.base_model:
				raise NotImplementedError("L2 loss does not work for decoder-only models yet")
			decoder_input_ids = [tokenizer.bos_token_id]
			labels = [int(label_str)]

		data_inputs.append(input_ids)
		data_labels.append(labels)
		if decoder_input_ids is not None:
			data_decoder_inputs.append(decoder_input_ids)
	# with open(filename, 'wb') as f:
	# 	if len(data_decoder_inputs) > 0:
	# 		pkl.dump((data_inputs, data_labels, data_decoder_inputs), f)
	# 		return data_inputs, data_labels, data_decoder_inputs
	# 	else:
	# 		pkl.dump((data_inputs, data_labels), f)
	# 		return data_inputs, data_labels
	if len(data_decoder_inputs) > 0:
		return data_inputs, data_labels, data_decoder_inputs
	else:
		return data_inputs, data_labels

def read_data(params, tokenizer):
	print('Loading data...')
	data = {'train': {'raw': [], 'tokenized': ([], [], [])}, 'val': {'raw': [], 'tokenized': ([], [], [])}, 'test': {'raw': [], 'alg': []}}
	path = f'{params.data_dir}/{params.dataset}'
	for split in ['train', 'val']:
		files = params.val_files if split == 'val' else params.train_files
		for file in files:
			filename = f'{path}/{split}/supervised_{file}.pkl'
			with open(filename, 'rb') as f:
				datapoints = pkl.load(f)
				data[split]['raw'].extend(datapoints)
			
			tokenized_data = tokenize_data(params, datapoints, tokenizer, filename)
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