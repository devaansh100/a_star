from .utils import *
from .AStar import AStar, Node
import scipy
import random

class Node_sokoban(Node):
	def __init__(self, player_pos, boxes, g, h =  0):
		super().__init__(player_pos, g, h)
		self.boxes = boxes

	@property
	def info(self):
		return self.pos, self.boxes
		
	def is_equal(self, node):
		return self.boxes == node.boxes and self.pos == node.pos

class AStar_sokoban(AStar):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def initialise_state(self, *args, **kwargs):
		self.puzzle, player, boxes, self.docks = convert_sb_to_array(args[0])
		if 'subsample' in kwargs:
			boxes = set(random.sample(boxes, kwargs['subsample']))
			self.docks = random.sample(self.docks, kwargs['subsample'])
		self.backlogged_node = Node_sokoban(player, boxes, 0)
		self.frontier = [self.backlogged_node]
		self.initial_boxes = boxes
		self.initial_pos = player

	def h(self, player_pos, boxes):
		undocked_boxes = []
		empty_docks = []
		for box in boxes:
			if box not in self.docks:
				undocked_boxes.append(box)
		for dock in self.docks:
			if dock not in boxes:
				empty_docks.append(dock)
		if len(empty_docks) > 1:
			G = np.zeros((len(empty_docks), len(undocked_boxes)))
			for i in range(len(empty_docks)):
				for j in range(len(undocked_boxes)):
					G[i, j] = manhattan_distance(empty_docks[i], undocked_boxes[j])
			d, b = scipy.optimize.linear_sum_assignment(G)
			h = sum([G[i, j] for i, j in zip(d, b)])
		else:
			h = manhattan_distance(empty_docks[0], undocked_boxes[0]) if len(empty_docks) == 1 else 0
		# NOTE: Strictly, you should subtract 1 from here since the player and box can never be at the same position. But this shouldn't have significantly different effects
		h += min([manhattan_distance(player_pos, box) for box in boxes])

		return int(h)

	def display(self):
		print('Frontier:')
		for x in self.frontier:
			print(convert_array_to_sb(self.puzzle, self.docks, x.boxes, x.pos))
			print(str((x.g, x.h)) + '\n')
		print('Closed:')
		for x in self.closed:
			print(convert_array_to_sb(self.puzzle, self.docks, x.boxes, x.pos))
			print(str((x.g, x.h)) + '\n')

	def show(self):
		for x in self.optimal_plan:
			print(convert_array_to_sb(self.puzzle, self.docks, x.boxes, x.pos) + '\n')

	def expand(self, node):
		x, y = node.pos
		boxes = node.boxes
		directions = (1, 0), (-1, 0), (0, 1), (0, -1)
		children = []

		for next_x, next_y in directions:
			child_pos = (x + next_x, y + next_y)
			if 0 <= child_pos[0] < self.size[0] and 0 <= child_pos[1] < self.size[1]:
				if self.puzzle[child_pos] == 0:
					if child_pos not in boxes:
						children.append(Node_sokoban(child_pos, boxes.copy(), g = node.g + 1))
					else:
						beside_child_pos = (x + 2 * next_x, y + 2 * next_y)
						no_box_beside_child = beside_child_pos not in node.boxes
						no_edge_beside_child = 0 <= beside_child_pos[0] < self.size[0] and 0 <= beside_child_pos[1] < self.size[1]
						no_wall_beside_child = self.puzzle[beside_child_pos] == 0
						if no_box_beside_child and no_edge_beside_child and no_wall_beside_child:
							new_boxes = boxes.copy()
							new_boxes.remove(child_pos)
							new_boxes.add(beside_child_pos)
							children.append(Node_sokoban(child_pos, new_boxes, g = node.g + 1))
		children = self.populate_h(children)
		return children
	
	def check_goal(self, node):
		return all([box in self.docks for box in node.boxes])
