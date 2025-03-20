from .cube import Cube
import random
import numpy as np

class Agent:

    def __init__(self):
        self.epsilon = 0.1 # exploration rate
        self.gamma = 0.9 # discount rate
        faces, directions =  np.meshgrid([0, 1, 2, 3, 4, 5], ['cc', 'cw', 'hf'])
        self.actions = list(zip(faces.flatten(), directions.flatten()))

    def get_all_states(self, cubes, n):
        '''recursively find all possible states of the cube after n moves'''

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

        # if n is 1, return the new cubes
        if n == 1:
            return new_cubes
        else:
            # recursively find all possible states for n-1 moves
            return self.get_all_states(new_cubes, n-1)
        
    def initialize_npz(self, n):
        '''
        generate data for n moves from the solved state and an 
        equal number of random states and save it to a npz file
        '''

        # find the total number of states
        total_states = 0
        for i in range(n):
            total_states += len(self.actions)**(i+1)

        assert total_states < 111151, f"Too many states to generate: {total_states}. Please reduce n."

        # initialize the output array
        X = np.zeros((total_states*2, 288), dtype=bool)
        y = np.zeros((total_states*2, 1), dtype=float)
        index = 0

        # generate the data for known states
        for i in range(n):
            state_value = self.gamma**(i+1)
            cubes = self.get_all_states(Cube(), i+1)
            for cube in cubes:
                X[index] = cube.flat_state()
                y[index] = state_value
                index += 1

        # generate random states and assign a value of 0
        n_scrambled = 20
        for _ in range(total_states):
            random_cube = Cube()
            for _ in range(random.randint(1, n_scrambled)):
                action = random.choice(self.actions)
                random_cube(action[0], action[1])
            X[index] = random_cube.flat_state()
            y[index] = 0
            index += 1

        # save X and y to an npz file
        np.savez("data.npz", X=X, y=y)