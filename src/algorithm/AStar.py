import heapq
import numpy as np
import torch
from .utils import *
import bisect

class Node():
	def __init__(self, pos, g, h = 0):
		self.pos = pos
		self.g = g
		self.h = h
		self.parent = None
		self.children = []
		self.is_optimal = False
		self.frontier = None
		self.frontier_snapshots = []
		self.deception_score = 0

	@property
	def cost(self):
		return self.g + self.h

	@property
	def info(self):
		return self.pos
	
	def __lt__(self, other):
		return self.cost < other.cost
	
	def is_equal(self, node):
		return self.pos == node.pos


class AStar():
	def __init__(self, *args, **kwargs):
		self.initialise_state(*args, **kwargs)
		self.enable_bar = kwargs['enable_bar'] if 'enable_bar' in kwargs else False
		self.size = self.puzzle.shape
		# heapq.heapify(self.frontier)
		self.closed = []
		self.optimal_plan = None
		self.terminate_after = kwargs['terminate_after'] if 'terminate_after' in kwargs else float('inf')
		self.frontier_snapshots = []
	
	def select(self):
		# node = heapq.heappop(self.frontier)
		node = self.frontier.pop(0)
		self.closed.append(node)
		return node
	
	def plan(self, node):
		optimal_plan = []
		while node != None:
			node.is_optimal = True
			optimal_plan.append(node)
			node = node.parent
		self.optimal_plan = optimal_plan[::-1][1:]
		# self.optimal_plan.pop(0)
	
	def populate_h(self, nodes):
		if self.backlogged_node is not None:
			self.backlogged_node.h = self.h(*self.backlogged_node.info)
			self.backlogged_node = None
		for i in range(len(nodes)):
			nodes[i].h = self.h(*nodes[i].info)
		return nodes
	
	def search(self):
		self.iterations = 0
		while len(self.frontier) > 0 and self.terminate_after > 0:
			self.frontier_snapshots.append(self.frontier.copy())
			self.iterations += 1
			self.terminate_after -= 1
			
			node = self.select()
			if self.check_goal(node):
				self.plan(node)
				break
			children = self.expand(node)
			for child in children:
				discard_child = False
				for prev_node in self.frontier + self.closed:
					if prev_node.is_equal(child) and prev_node.cost <= child.cost:
						discard_child = True
						break
				if discard_child:
					continue
				child.parent = node
				node.children.append(child)
				# heapq.heappush(self.frontier, child)
				bisect.insort(self.frontier, child)
			# if self.iterations % 1000 == 0:
			# 	print(f'{self.iterations=}')

def get_improved_heuristic_solver(solver):
	class ModelAStar(solver):
		def __init__(self, *args, **kwargs):
			self.model = kwargs['model']
			self.prompt = kwargs['prompt'].replace('{initial_str}', args[0])
			self.domain = kwargs['domain']
			# self.kv_cache = kwargs['kv_cache']
			self.device = kwargs['device']
			self.checked_prompts = {}
			self.num_return_sequences = kwargs['num_return_sequences']
			super().__init__(*args, **kwargs)
		
		def populate_h(self, nodes):
			if self.device == 'cpu':
				return super().populate_h(nodes)
			if self.backlogged_node is not None:
				nodes += [self.backlogged_node]
			prompts = []
			heuristics = []
			prev_checked = set()
			for i in range(len(nodes)):
				heuristics.append(super().h(*nodes[i].info))
				prompt = self.create_prompt(*nodes[i].info, heuristics[-1])
				if prompt not in self.checked_prompts:
					prompts.append(prompt)
				else:
					prev_checked.add(i)
					nodes[i].h = heuristics[-1] + self.checked_prompts[prompt]
			
			differences = self.get_difference(prompts)
			for i in range(len(nodes)):
				if i not in prev_checked:
					difference = differences.pop(0)
					nodes[i].h = heuristics[i] + difference
					self.checked_prompts[prompts.pop(0)] = difference
			if self.backlogged_node is not None:
				nodes.pop() # Pop so it not conisdered in the children
				self.backlogged_node = None
			return nodes

		def create_prompt(self, *args):
			if self.domain == 'maze':
				puzzle_str = convert_array_to_maze(self.puzzle, args[0], self.goal)
			else:
				puzzle_str = convert_array_to_sb(self.puzzle, self.docks, args[1], args[0])
			prompt = self.prompt.replace('{puzzle_str}', puzzle_str).replace('{heuristic}', str(args[-1]))
			return prompt

		def h(self, *args): # Only used for CPU inference
			h = super().h(*args)
			prompt = self.create_prompt(*args, h)
			if prompt not in self.checked_prompts:
				difference = self.get_difference(prompt)[0]
				self.checked_prompts[prompt] = difference
			else:
				difference = self.checked_prompts[prompt]
			return h + difference
				
		def get_difference(self, prompts):
			difference = []
			if len(prompts) > 0:
				model_inputs = self.model.inference_tokenizer(prompts, return_tensors='pt', padding = True)
				model_inputs = {k: v.to(torch.device(self.device)) for k, v in model_inputs.items()}
				output = self.model.model.generate(**model_inputs, do_sample = True, top_k = 5, num_return_sequences = self.num_return_sequences, max_new_tokens = 5, pad_token_id = self.model.inference_tokenizer.eos_token_id) #, past_key_values = self.kv_cache[len(prompts)])
				output_text = self.model.inference_tokenizer.batch_decode(output, skip_special_tokens=True)
				difference = extract_differences(output_text)
				difference = [difference[i : i + 3] for i in range(0, len(difference), 3)]
				difference = [max(set(x), key=x.count) for x in difference]
				# print(prompt)
				# print(difference)
				# breakpoint()
			return difference

	return ModelAStar

def get_random_heuristic_solver(solver):
	class ModelAStar(solver):
		def __init__(self, *args, **kwargs):
			kwargs['domain']
			self.strategy = kwargs['strategy']
			if kwargs['domain'] == 'maze': # 5/7/10 maze
				self.diffs = np.array([0, 2, 6, 4, 8, 10, 12, 14, 22, 16, 18, 20])
				self.weights = torch.softmax(torch.tensor([5000, 5000, 5000, 5000, 4875, 2232, 885, 306, 1, 101, 16, 2]).float(), dim = 0).numpy()
			elif kwargs['domain'] == 'sokoban':
				self.diffs = torch.tensor([7, 12, 13, 22, 10, 9, 6, 5, 11, 15, 14, 3, 18, 16, 8, 2, 1, 4, 17, 20, 19, 21, 23, 27, 29, 0, 24, 25, 28, 26, 30, 32, 36, 35, 31, 33, 40])
				self.weights = torch.softmax(torch.tensor([3099, 1396, 1380, 81, 2008, 2962, 2652, 2997, 2046, 808, 847, 1688, 371, 631, 2604, 450, 118, 2175, 463, 201, 237, 126, 79, 17, 7, 9, 43, 46, 15, 21, 5, 2, 1, 1, 4, 2, 1]).float(), dim = 0).numpy()
			super().__init__(*args)
		
		def h(self, *args):
			h = super().h(*args)
			if self.strategy == 'uniform':
				difference = np.random.choice(self.diffs)
			elif self.strategy == 'td':
				difference = np.random.choice(self.diffs, p = self.weights)
			# breakpoint()
			return h + difference

	return ModelAStar
