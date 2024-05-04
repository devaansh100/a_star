import numpy as np
import torch

def convert_array_to_maze(puzzle, start, goal):
	maze_str = ''
	for r in range(len(puzzle)):
		for c in range(len(puzzle[0])):
			if puzzle[r, c] == 1:
				maze_str += '#'
			else:
				if (r, c) == start:
					maze_str += '@'
				elif (r, c) == goal:
					maze_str += 'X'
				else:
					maze_str += '.'
		maze_str += '\n'
	return maze_str.strip()

def convert_maze_to_array(puzzle_str):
	puzzle = [[]]
	for ch in puzzle_str:
		if ch == '.':
			puzzle[-1].append(0)
		elif ch == '#':
			puzzle[-1].append(1)
		elif ch == '@':
			puzzle[-1].append(0)
			start = (len(puzzle) - 1, len(puzzle[-1]) - 1)
		elif ch == 'X':
			puzzle[-1].append(0)
			goal = (len(puzzle) - 1, len(puzzle[-1]) - 1)
		elif ch == '\n':
			puzzle.append([])
	return np.array(puzzle), start, goal


def convert_sb_to_array(puzzle_str):
	puzzle = np.zeros((10, 10))
	boxes, docks = set(), []
	r, c = 0, 0
	for ch in puzzle_str:
		if ch == '#':
			puzzle[r, c] = 1
		elif ch == '$':
			boxes.add((r, c))
		elif ch == '.':
			docks.append((r, c))
		elif ch == '@':
			player = (r, c)
		elif ch == 'O':
			player = (r, c)
			docks.append((r, c))
		elif ch == 'X':
			boxes.add((r, c))
			docks.append((r, c))

		if ch == '\n':
			r += 1
			c = 0
		else:
			c += 1
	return puzzle, player, boxes, docks

def convert_array_to_sb(puzzle, docks, boxes, player):
	puzzle_str = ''
	for r, row in enumerate(puzzle):
		puzzle_str += '\n'
		for c, ele in enumerate(row):
			if ele == 1:
				ch = '#'
			else:
				if (r, c) in docks:
					if (r, c) not in boxes:
						if (r, c) == player:
							ch = 'O'
						else:
							ch = '.'
					else:
						ch = 'X'
				else:
					if (r, c) in boxes:
						ch = '$'
					elif (r, c) == player:
						ch = '@'
					else:
						ch = ' '
			puzzle_str += ch
	return puzzle_str.strip()

def manhattan_distance(pos1, pos2):
	return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def chebyshev_distance(pos1, pos2):
	return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

def extract_differences(diffs):
	# diffs = [output.split('difference =')[-1].split('<|endoftext|>')[0] for output in outputs] # NOTE: Only useful for decoder-only models
	for i in range(len(diffs)):
		try:
			diffs[i] = int(diffs[i])
		except:
			vals = diffs[i].split(',')
			try:
				if len(vals) > 1:
					vals[1] = vals[1].split('=')[-1]
			except:
				vals[1] = '0'
			for j in range(len(vals)):
				try:
					vals[j] = int(vals[j])
				except:
					vals[j] = 0
			diffs[i] = sum(vals)
	return diffs