from .utils import chebyshev_distance, convert_array_to_maze, convert_maze_to_array, manhattan_distance
from .AStar import AStar, Node
import math
import heapq
import numpy as np

class AStar_maze(AStar):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def initialise_state(self, *args, **kwargs):
		self.puzzle, self.start, self.goal = convert_maze_to_array(args[0])
		self.backlogged_node = Node(self.start, g = 0)
		self.frontier = [self.backlogged_node]

	def h(self, node):
		x, y = node.info
		return manhattan_distance((x, y), self.goal)

	def show(self):
		print(convert_array_to_maze(self.puzzle, self.start, self.goal))
		print([x.pos for x in self.optimal_plan])

	def expand(self, node):
		x, y = node.pos
		next_nodes = (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)
		children = []

		for child_pos in next_nodes:
			if 0 <= child_pos[0] < self.size[0] and 0 <= child_pos[1] < self.size[1]:
				if self.puzzle[child_pos] == 0:
					children.append(Node(child_pos, g = node.g + 1))
		children = self.populate_h(children)
		return children

	def show_lists(self):
		frontier = [x.info for x in self.frontier]
		closed = [x.info for x in self.closed]
		print(f'Frontier: {frontier}')
		print(f'Closed: {closed}')
	
	def check_goal(self, node):
		return node.pos == self.goal

class Dijkstra_maze():
	def __init__(self, *args, **kwargs):
		self.initialise_state(*args, **kwargs)
		self.size = self.puzzle.shape
	
	def initialise_state(self, *args, **kwargs):
		self.puzzle, self.start, self.goal = convert_maze_to_array(args[0])
	
	def show(self):
		print(convert_array_to_maze(self.puzzle, self.start, self.goal))
		
	def fill(self, from_start):
	    starting_pos = self.start if from_start else self.goal
	    # Define the movement directions
	    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
			
	    # Create a distance matrix to store the minimum distance from the start
	    dist = np.array([[float('inf')] * self.size[1] for _ in range(self.size[0])])
	    dist[starting_pos] = 0
		
	    # Create a priority queue and add the start node
	    pq = [(0, starting_pos)]
		
	    # Run Dijkstra's algorithm
	    while pq:
	        # Pop the node with the minimum distance from the priority queue
	        curr_dist, curr_node = heapq.heappop(pq)
			
	        # Skip if the current node has been visited
	        if dist[curr_node] < curr_dist:
	            continue
			
	        # Explore the neighbors of the current node
	        for dx, dy in directions:
	            nx, ny = curr_node[0] + dx, curr_node[1] + dy
				
	            # Skip if the neighbor is out of bounds or is a wall
	            if (0 <= nx < self.size[0] and 0 <= ny < self.size[1]) and (self.puzzle[nx, ny] == 0):
	                # Update the distance if a shorter path is found
	                neighbor_dist = curr_dist + 1
	                if neighbor_dist < dist[nx, ny]:
	                    dist[nx, ny] = neighbor_dist
	                    heapq.heappush(pq, (neighbor_dist, (nx, ny)))
	    return dist

class Oracle_maze(AStar_maze):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		dijkstra = Dijkstra_maze(*args, **kwargs)
		self.perfect_heuristic = dijkstra.fill(from_start = False)
		self.type = kwargs['type']
		self.ref_alg = kwargs['astar']
		# assert self.type in ['perfect', 'easy', 'med', 'hard']
		self.num_perf = 0
		if self.type != 'perfect':
			self.std = kwargs['std']
			if 'astar' in kwargs:
				astar = kwargs['astar']
			else:
				astar = AStar_maze(*args, **kwargs)
				astar.search()
			# diffs = [len(astar.optimal_plan) - node.g - node.h for node in astar.optimal_plan]
			# self.min_diff, self.max_diff = min(diffs), max(diffs)
			hard = set(range(0, len(astar.optimal_plan) // 3))
			med = set(range(len(astar.optimal_plan) // 3, 2 * len(astar.optimal_plan) // 3))
			easy = set(range(2 * len(astar.optimal_plan) // 3, len(astar.optimal_plan)))
			if self.type == 'easy':
				self.perfect_sets = easy
			elif self.type == 'med':
				self.perfect_sets = med
			elif self.type == 'hard':
				self.perfect_sets = hard
			elif isinstance(self.type, int):
				self.perfect_sets = set([self.type])
		
	def sample_diff(self, node, h):
		h_star = self.perfect_heuristic[node.pos]
		diff = h_star - h
		gaussian_eqn = lambda x, mean, std : np.exp(-0.5 * ((x - mean)/std)**2)/(std * np.sqrt(2 * np.pi))
		possible_diffs = list(range(self.min_diff, self.max_diff + 1))
		w = np.array([gaussian_eqn(d, diff, self.std) for d in possible_diffs])
		w /= w.sum()
		return np.random.choice(possible_diffs, p = w, size = 1)
	
	def h(self, node):
		h_star = self.perfect_heuristic[node.pos]
		# if self.type == 'perfect' or node.g in self.perfect_sets:
		# 	h = h_star
		# 	self.num_perf += 1
		# else:
		h = np.random.normal(loc = h_star, scale = self.std, size = 1)
		return h
