from .utils import *
from .AStar import AStar, Node
import scipy
import random

class Node_stp(Node):
	def __init__(self, board_pos, g, h =  0):
		super().__init__(board_pos, g, h)
		self.empty_pos = np.nonzero(self.pos == 0)
		self.str_state = convert_array_to_stp(board_pos)

	@property
	def info(self):
		return self.pos
		
	def is_equal(self, node):
		return self.str_state == node.str_state

class AStar_stp(AStar):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def initialise_state(self, *args, **kwargs):
		puzzle = convert_stp_to_array(args[0])
		self.backlogged_node = Node_stp(puzzle, 0)
		self.frontier = [self.backlogged_node]
		self.size = puzzle.shape
		self.goal = np.zeros(self.size, dtype = np.int32)
		self.goal_dict = {}
		goal_number = iter(range(self.size[0] * self.size[1]))
		for i in range(self.size[0]):
			for j in range(self.size[1]):
				g = next(goal_number)
				self.goal[i, j] = g
				self.goal_dict[g] = (i, j)
		self.goal = convert_array_to_stp(self.goal)

	def h(self, node):
		# if node.parent is None:
		h = 0
		for i in range(self.size[0]):
			for j in range(self.size[1]):
				h += manhattan_distance((i, j), self.goal_dict[node.pos[i, j]])
		# else:
		# 	x1, y1 = node.empty_pos
		# 	x2, y2 = node.parent.empty_pos
		# 	# For parent, (x2, y2) is blank and (x1, y1) is num
		# 	# For child, (x1, y1) is blank and (x2, y2) is num
		# 	assert (node.parent.pos[x1, y1] == node.pos[x2, y2]) and (node.parent.pos[x2, y2] == node.pos[x1, y1] == 0)
		# 	num = node.pos[x2, y2].item()
		# 	h = node.parent.h - manhattan_distance((x1, y1), self.goal_dict[num]) - manhattan_distance((x2, y2), self.goal_dict[0]) \
		# 					  + manhattan_distance((x1, y1), self.goal_dict[0]) + manhattan_distance((x2, y2), self.goal_dict[num])
		return h

	def display(self):
		print('Frontier:')
		for x in self.frontier:
			print(convert_array_to_stp(x.pos))
			print(str((x.g, x.h)) + '\n')
		print('Closed:')
		for x in self.closed:
			print(convert_array_to_stp(x.pos))
			print(str((x.g, x.h)) + '\n')

	def show(self):
		for x in self.optimal_plan:
			print(convert_array_to_stp(x.pos) + '\n')

	def expand(self, node):
		x, y = node.empty_pos
		directions = (1, 0), (-1, 0), (0, 1), (0, -1)
		children = []
		for next_x, next_y in directions:
			child_pos = (x + next_x, y + next_y)
			if 0 <= child_pos[0] < self.size[0] and 0 <= child_pos[1] < self.size[1]:
				child_puzzle = node.pos.copy()
				child_puzzle[child_pos], child_puzzle[node.empty_pos] = child_puzzle[node.empty_pos], child_puzzle[child_pos]
				child = Node_stp(child_puzzle, g = node.g + 1)
				child.parent = node # NOTE: This is being overwritten by the same value later. It is set here once to perform incremental calculations forh
				children.append(child)
		children = self.populate_h(children)
		return children
	
	def check_goal(self, node):
		return node.str_state == self.goal