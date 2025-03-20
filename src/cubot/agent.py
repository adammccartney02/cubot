from .cube import Cube
import random
import numpy as np

class Agent:

    def __init__(self):
        self.epsilon = 0.1
        faces, directions =  np.meshgrid([0, 1, 2, 3, 4, 5], ['cc', 'cw', 'hf'])
        self.actions = list(zip(faces.flatten(), directions.flatten()))

    def get_all_states(self, cubes, n):
        '''
        recursively find all possible states of the cube after n moves
        '''
        # if cubes is a single cube, convert it to a list
        if isinstance(cubes, Cube):
            cubes = [cubes]

        # find all possible next states
        new_cubes = []
        for cube in cubes:
            for action in self.actions:
                new_cube = cube.copy()
                new_cube(action[0], action[1])
                new_cubes.append(new_cube)

        if n == 1:
            return new_cubes
        else:
            return self.get_all_states(new_cubes, n-1)