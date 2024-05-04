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
import sys
sys.path.append('data/mazelib')
# sys.path.append('mazelib/mazelib/generate')
from mazelib import Maze
from mazelib.generate.Prims import Prims

# def generate_maze(size):
# 	maze = np.zeros((size, size))
# 	wall_percent = np.random.uniform(0.3, 0.5)

# 	# This guarantees 30 - 50% walls
# 	wall_number = int(wall_percent * size * size)
# 	wall_x = np.random.choice(np.arange(size), wall_number)
# 	wall_y = np.random.choice(np.arange(size), wall_number)
# 	maze[wall_x, wall_y] = 1

# 	open_pos = (1 - maze).nonzero()
# 	start, goal = np.random.choice(np.arange(len(open_pos[0])), 2, replace = False)
# 	start = open_pos[0][start], open_pos[1][start]
# 	goal = open_pos[0][goal], open_pos[1][goal]
# 	return convert_array_to_maze(maze, start, goal)
def generate_maze(width, height):
	m = Maze()
	m.generator = Prims(width, height)
	m.generate()
	return m


# def create_maze_dataset(params, num_train, num_val, num_test, size, solver = None):
# 	solver = AStar_maze if solver is None else solver
# 	puzzles = []
# 	p_bar = tqdm(range(num_train + num_val))
# 	mazes = set()
# 	while len(puzzles) < num_train + num_val:
# 		maze = generate_maze(size)
# 		alg = solver(maze)
# 		if maze not in mazes:
# 			mazes.add(maze)
# 			alg.search()
# 			if alg.optimal_plan is not None:
# 				if len(alg.optimal_plan) > size and len(alg.optimal_plan)/alg.closed[0].h > 1.3:
# 					puzzles.append((maze, alg))
# 					p_bar.n += 1
# 					p_bar.refresh()
	
# 	random.shuffle(puzzles)
# 	train, val, test = puzzles[:num_train], puzzles[num_train:], []
# 	p_bar = tqdm(range(num_test))
# 	while len(test) < num_test:
# 		maze = generate_maze(size)
# 		alg = AStar_maze(maze)
# 		if maze not in mazes:
# 			mazes.add(maze)
# 			alg.search()
# 			if alg.optimal_plan is not None:
# 				if len(alg.optimal_plan) > size and len(alg.optimal_plan)/alg.closed[0].h > 1.05:
# 					test.append((maze, alg))
# 					p_bar.n += 1
# 					p_bar.refresh()

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


# def create_maze_dataset(params, num_train, num_val, num_test, solver = None):
# 	solver = AStar_maze if solver is None else solver
# 	train_mazes = np.load(f'{params.data_dir}/maze-transpath/train/maps.npy')
# 	val_mazes = np.load(f'{params.data_dir}/maze-transpath/val/maps.npy')
# 	test_mazes = np.load(f'{params.data_dir}/maze-transpath/test/maps.npy')

# 	train_starts = np.load(f'{params.data_dir}/maze-transpath/train/starts.npy')
# 	val_starts = np.load(f'{params.data_dir}/maze-transpath/val/starts.npy')
# 	test_starts = np.load(f'{params.data_dir}/maze-transpath/test/starts.npy')

# 	train_goals = np.load(f'{params.data_dir}/maze-transpath/train/goals.npy')
# 	val_goals = np.load(f'{params.data_dir}/maze-transpath/val/goals.npy')
# 	test_goals = np.load(f'{params.data_dir}/maze-transpath/test/goals.npy')
	
# 	train_idxs = np.random.choice(np.arange(train_mazes.shape[0]), num_train, replace = False)
# 	val_idxs = np.random.choice(np.arange(val_mazes.shape[0]), num_val, replace = False)
# 	test_idxs = np.random.choice(np.arange(test_mazes.shape[0]), num_test, replace = False)

# 	puzzles = {'train': {}, 'val': {}, 'test': {}}
# 	puzzles['train']['mazes'], puzzles['train']['starts'], puzzles['train']['goals'] = train_mazes[train_idxs], train_starts[train_idxs], train_goals[train_idxs]
# 	puzzles['val']['mazes'], puzzles['val']['starts'], puzzles['val']['goals'] = val_mazes[val_idxs], val_starts[val_idxs], val_goals[val_idxs]
# 	puzzles['test']['mazes'], puzzles['test']['starts'], puzzles['test']['goals'] = test_mazes[test_idxs], test_starts[test_idxs], test_goals[test_idxs]
# 	output_files = {}
# 	for split in puzzles.keys():
# 		output_files[split] = {'mazes': [], 'algs': []}
# 		for (maze, start, goal) in tqdm(zip(puzzles[split]['mazes'], puzzles[split]['starts'], puzzles[split]['goals']), total = len(puzzles[split]['mazes'])):
# 			maze = maze[0].astype(np.int32)
# 			x, y = start[0].nonzero()
# 			start = int(x), int(y)

# 			x, y = goal[0].nonzero()
# 			goal = int(x), int(y)
# 			puzzle_str = convert_array_to_maze(maze, start, goal)
# 			alg = solver(puzzle_str)
# 			alg.search()
# 			if alg.optimal_plan is not None:
# 				output_files[split]['mazes'].append(puzzle_str)
# 				output_files[split]['algs'].append(alg)

# 	print(f"Train: {len(output_files['train']['mazes'])}, Val: {len(output_files['val']['mazes'])}, Test: {len(output_files['test']['mazes'])}")
# 	os.makedirs(f'{params.data_dir}/{params.dataset}/train', exist_ok = True)
# 	os.makedirs(f'{params.data_dir}/{params.dataset}/val', exist_ok = True)
# 	os.makedirs(f'{params.data_dir}/{params.dataset}/test', exist_ok = True)

# 	if num_train > 0:
# 		with open(f'{params.data_dir}/{params.dataset}/train/mazes.txt', 'w') as f:
# 			f.write(';'.join(output_files['train']['mazes']))
		
# 		with open(f'{params.data_dir}/{params.dataset}/train/alg_mazes.pkl', 'wb') as f:
# 			pkl.dump(output_files['train']['algs'], f)
	
# 	if num_val > 0:
# 		with open(f'{params.data_dir}/{params.dataset}/val/mazes.txt', 'w') as f:
# 			f.write(';'.join(output_files['val']['mazes']))
		
# 		with open(f'{params.data_dir}/{params.dataset}/val/alg_mazes.pkl', 'wb') as f:
# 			pkl.dump(output_files['val']['algs'], f)
	
# 	if num_test > 0:
# 		with open(f'{params.data_dir}/{params.dataset}/test/mazes.txt', 'w') as f:
# 			f.write(';'.join(output_files['test']['mazes']))

# 		with open(f'{params.data_dir}/{params.dataset}/test/alg_mazes.pkl', 'wb') as f:
# 			pkl.dump(output_files['test']['algs'], f)

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
	for node in alg.closed:
		num_generations = float('Inf')
		while not node.is_optimal and num_generations > 0:
			node.deception_score += 1
			num_generations -= 1
			node = node.parent

def create_supervision(params, solver = None):
	
	def random_sample(closed_set):
		return closed_set.pop(0)
	
	def deception_sample(closed_set):
		scores = torch.tensor([node.deception_score for node in closed_set], dtype = torch.float64)
		weights = torch.softmax(scores, -1)
		node = np.random.choice(closed_set, p = weights)
		closed_set.remove(node)
		return node
	
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
		sampled_nodes = params.sampled_nodes if split == 'train' else seqs_per_puzzle // 2
		if len(params.alg_files) > 0:
			alg_files = [f'{path}/{split}/{alg_file}.pkl' for alg_file in params.alg_files]
		alg_files = glob.glob(f'{path}/{split}/*.pkl')
		for alg_file in alg_files:
			try:
				if params.domain == 'sokoban':
					incorrect_file = str(params.create_data[3]) not in alg_file or 'supervised' in alg_file
				else:
					incorrect_file = 'supervised' in alg_file
			except:
				if params.domain == 'sokoban':
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
				if 'rand' in params.sample:
					random.shuffle(closed_set)
				annotate_deceptive_nodes(alg)
				num_chosen, iters = seqs_per_puzzle, 0
				while num_chosen > 0 and len(closed_set) > 0:
					if params.sample == 'random':
						node = random_sample(closed_set)

					elif 'optimal' in params.sample:
						nodes, optimal_costs = optimal_sample(alg, num_chosen, difficulty = params.sample.split('optimal_')[-1])
						num_chosen -= len(nodes)
						for node, optimal_cost in zip(nodes, optimal_costs):
							dataset.append((initial_str, get_puzzle_str(alg, node), node.h, optimal_cost, node.deception_score))
						node = nodes[0]

					elif params.sample == 'deception':
						node = deception_sample(closed_set)

					elif params.sample == 'opt_dec_rand':
						if num_chosen > (seqs_per_puzzle - sampled_nodes/2):
							node = random_sample(closed_set)
						elif num_chosen > (seqs_per_puzzle - sampled_nodes):
							node = deception_sample(closed_set)
						else:
							nodes, optimal_costs = optimal_sample(alg, num_chosen)
							num_chosen -= len(nodes)
							for node, optimal_cost in zip(nodes, optimal_costs):
								dataset.append((initial_str, get_puzzle_str(alg, node), node.h, optimal_cost, node.deception_score))
					
					elif 'rand_' in params.sample:
						if num_chosen > (seqs_per_puzzle - sampled_nodes): # seqs_per_puzzle // 2:
							node = random_sample(closed_set)
						else:
							if 'opt' in params.sample: # rand_opt
								nodes, optimal_costs = optimal_sample(alg, num_chosen)
								num_chosen -= len(nodes)
								for node, optimal_cost in zip(nodes, optimal_costs):
									dataset.append((initial_str, get_puzzle_str(alg, node), node.h, optimal_cost, node.deception_score))
							elif 'dec' in params.sample: # rand_dec
								node = deception_sample(closed_set)
							
					elif params.sample == 'opt_dec':
						if num_chosen > (seqs_per_puzzle - sampled_nodes): # seqs_per_puzzle // 2:
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
							optimal_cost = new_alg.optimal_plan[-1].g # len(new_alg.optimal_plan)
							dataset.append((initial_str, puzzle_str, node.h, optimal_cost, node.deception_score))
							num_chosen -= 1
							iters = 0
					iters += 1
			sample = params.sample
			if params.sample == 'rand_opt':
				sample = f'rand{params.sampled_nodes}_opt'
			elif params.sample == 'opt_dec':
				sample = f'opt_dec{params.sampled_nodes}'
			alg_file = alg_file.split('/')[-1].replace('.pkl', f'_{sample}.pkl')
			with open(f'{path}/{split}/supervised_{alg_file}', 'wb') as f:
				pkl.dump(dataset, f)
			print(f'Created {len(dataset)}')

def tokenize_data(params, datapoints, tokenizer, filename):
	legend = {'maze': "@ - player, # - wall, . - empty cell, X - goal", 'sokoban': "@ - player, # - wall, . - empty docks, ' ' - empty cell, $ - box, X - box on dock, O - player on dock"}
	data_inputs, data_labels, data_decoder_inputs = [], [], []
	with open(params.prompt_file) as f:
		prompt = f.read()
	for datapoint in tqdm(datapoints, desc = filename):
		if len(datapoint) == 3:
			puzzle_str, heuristic, optimal_cost = datapoint
		elif len(datapoint) == 5:
			initial_str, puzzle_str, heuristic, optimal_cost, deception_score = datapoint
		if isinstance(heuristic, tuple):
			heuristic, pred_diff = heuristic
		else:
			pred_diff = None
		# initial_str, deception_score = '', 0
		# optimal_cost is actually counting one extra step, since start and goal are in the plan for maze. In sokoban, you don't end at the goal
		if params.target == 'dec':
			difference = deception_score
		else:
			difference = optimal_cost - heuristic
		input_prompt = prompt.replace('{puzzle_str}', puzzle_str).replace('{heuristic}', str(int(heuristic))).replace('{initial_str}', initial_str)
		input_prompt = input_prompt.replace('{puzzle_legend}', legend[params.domain]).replace('{domain}', params.domain)
		input_ids = tokenizer(input_prompt).input_ids
		if isinstance(difference, int):
			label_str = f'{int(difference)}'
		else:
			label_str = f'{round(difference, 2)}'
		if pred_diff is not None:
			label_str = f'{int(pred_diff)}, error = {optimal_cost - heuristic - pred_diff}'
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
	if is_main_process():
		print('Loading data...')
	data = {'train': {'raw': [], 'tokenized': ([], [], [])}, 'val': {'raw': [], 'tokenized': ([], [], [])}, 'test': {'raw': [], 'alg': []}}
	path = f'{params.data_dir}/{params.dataset}'
	for split in ['train', 'val']:
		files = params.val_files if split == 'val' else params.train_files
		for file in files:
			with open(f'{path}/{split}/supervised_{file}.pkl', 'rb') as f:
				datapoints = pkl.load(f)
				data[split]['raw'].extend(datapoints)
			
			if params.target == 'dec':
				tokenized_file = f"{params.data_dir}/{params.dataset}/{split}/{params.base_model}/tokenized_{file}_dec.pkl"
			else:
				tokenized_file = f"{params.data_dir}/{params.dataset}/{split}/{params.base_model}/tokenized_{file}.pkl"
			if os.path.exists(tokenized_file):
				with open(tokenized_file, 'rb') as f:
					tokenized_data = pkl.load(f)
			else:
				os.makedirs(f"{params.data_dir}/{params.dataset}/{split}/{params.base_model}", exist_ok = True)
				tokenized_data = tokenize_data(params, datapoints, tokenizer, tokenized_file)
			data[split]['tokenized'][0].extend(tokenized_data[0])
			data[split]['tokenized'][1].extend(tokenized_data[1])
			if len(tokenized_data) == 3:
				data[split]['tokenized'][2].extend(tokenized_data[2])
		if len(data[split]['tokenized'][-1]) == 0:
			data[split]['tokenized'] = data[split]['tokenized'][:-1]
	
	ilr_split = params.test_ilr[0]
	for file in params.test_ilr[1:]:
		with open(f'{path}/{ilr_split}/{file}.txt') as f:
			data['test']['raw'].extend(f.read().split(';'))
		
		with open(f"{path}/{ilr_split}/alg_{file}.pkl", 'rb') as f:
			data['test']['alg'].extend(pkl.load(f))
	return data