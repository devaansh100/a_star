from .utils import chebyshev_distance, convert_array_to_maze, convert_maze_to_array
from .AStar import AStar, Node
import math
# class AStar_maze(AStar):
# 	def __init__(self, *args, **kwargs):
# 		super().__init__(*args, **kwargs)
	
# 	def initialise_state(self, *args, **kwargs):
# 		self.puzzle, self.start, self.goal = convert_maze_to_array(args[0])
# 		self.backlogged_node = Node(self.start, g = 0)
# 		self.frontier = [self.backlogged_node]

# 	def h(self, x, y):
# 		return manhattan_distance((x, y), self.goal)

# 	def show(self):
# 		print(convert_array_to_maze(self.puzzle, self.start, self.goal))
# 		print([x.pos for x in self.optimal_plan])

# 	def expand(self, node):
# 		x, y = node.pos
# 		next_nodes = (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)
# 		children = []

# 		for child_pos in next_nodes:
# 			if 0 <= child_pos[0] < self.size[0] and 0 <= child_pos[1] < self.size[1]:
# 				if self.puzzle[child_pos] == 0:
# 					children.append(Node(child_pos, g = node.g + 1))
# 		children = self.populate_h(children)
# 		return children

# 	def show_lists(self):
# 		frontier = [x.info for x in self.frontier]
# 		closed = [x.info for x in self.closed]
# 		print(f'Frontier: {frontier}')
# 		print(f'Closed: {closed}')
	
# 	def check_goal(self, node):
# 		return node.pos == self.goal

class AStar_maze(AStar):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def initialise_state(self, *args, **kwargs):
		self.puzzle, self.start, self.goal = convert_maze_to_array(args[0])
		self.backlogged_node = Node(self.start, g = 0)
		self.frontier = [self.backlogged_node]

	def h(self, x, y):
		return chebyshev_distance((x, y), self.goal)

	def show(self):
		print(convert_array_to_maze(self.puzzle, self.start, self.goal))
		print([x.pos for x in self.optimal_plan])

	def expand(self, node):
		x, y = node.pos
		next_nodes = (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)
		children = []

		for i, child_pos in enumerate(next_nodes):
			if 0 <= child_pos[0] < self.size[0] and 0 <= child_pos[1] < self.size[1]:
				if self.puzzle[child_pos] == 0:
					cost = math.sqrt(2) if i > 3 else 1
					children.append(Node(child_pos, g = node.g + cost))
		children = self.populate_h(children)
		return children

	def show_lists(self):
		frontier = [x.info for x in self.frontier]
		closed = [x.info for x in self.closed]
		print(f'Frontier: {frontier}')
		print(f'Closed: {closed}')
	
	def check_goal(self, node):
		return node.pos == self.goal
