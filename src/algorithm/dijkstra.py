from collections import defaultdict
import heapq
from .utils import convert_array_to_maze, convert_maze_to_array
import numpy as np

class Dijkstra_Maze():
	def __init__(self, *args, **kwargs):
		self.initialise_state(*args, **kwargs)
		self.enable_bar = kwargs['enable_bar'] if 'enable_bar' in kwargs else False
		self.size = self.puzzle.shape
	
	def initialise_state(self, *args, **kwargs):
		self.puzzle, self.start, self.goal = convert_maze_to_array(args[0])
	
	def show(self):
		print(convert_array_to_maze(self.puzzle, self.start, self.goal))
		
	def dijkstra(self, from_start = True):
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