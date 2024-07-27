from .utils import *
from .AStar import AStar, Node
import scipy
import random

class Node_stp(Node):
	def __init__(self, board_pos, g, h =  0):
		super().__init__(board_pos, g, h)
		self.empty_pos = np.nonzero(self.pos == '0')
		self.str_state = convert_array_to_stp(board_pos)

	@property
	def info(self):
		return (self.pos,)
		
	def is_equal(self, node):
		return self.str_state == node.str_state

class AStar_stp(AStar):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def initialise_state(self, *args, **kwargs):
		puzzle_str = args[0]
		puzzle = convert_stp_to_array(puzzle_str)
		self.backlogged_node = Node_stp(puzzle, 0)
		self.frontier = [self.backlogged_node]
		self.size = puzzle.shape
		self.goal = convert_stp_to_array(get_stp_goal(puzzle))
		self.goal_dict = {self.goal[i, j]: (i, j) for j in range(self.size[1]) for i in range(self.size[0])}
		self.goal = convert_array_to_stp(self.goal)

	def h(self, node):
		h = 0
		for i in range(self.size[0]):
			for j in range(self.size[1]):
				if node.pos[i, j] != '0':
					h += manhattan_distance((i, j), self.goal_dict[node.pos[i, j]])
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
				children.append(child)
		children = self.populate_h(children)
		return children
	
	def check_goal(self, node):
		return node.str_state == self.goal