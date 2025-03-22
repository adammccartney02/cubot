from .cube import Cube
from .value import ValueFunction
import random
import numpy as np

class Agent:

    # type alias
    type Xy = tuple[np.ndarray, np.ndarray]
    type Cuboid = list[Cube]|Cube
    type Act = list[int, str]

    def __init__(self, hidden_shape:list[int]):
        # set hyperparameters
        self.epsilon = 0.1 # exploration rate
        self.gamma = 0.9 # discount rate

        # set actions
        faces, directions =  np.meshgrid([0, 1, 2, 3, 4, 5], ['cc', 'cw', 'hf'])
        self.actions = list(zip(faces.flatten(), directions.flatten()))

        # build model
        self.model = ValueFunction(hidden_shape=hidden_shape)

    def __call__(self, cube:Cube):
        return self.model(cube)
    
    def train(self, start, stop, step, epochs=1_000):
        # n moves
        X_known, y_known = self.n_step_data(start)

        # scrambled data
        X_scram, y_scram = self.scrambled_like(X_known)

        # combine and shuffle
        X, y = np.vstack((X_known, X_scram)), np.vstack((y_known, y_scram))
        p = np.random.permutation(len(X))
        X, y = X[p], y[p]

        # first training
        self.model.train_vf(X, y, epochs=epochs)


    def n_step_states(self, cubes:Cuboid, n=1) -> Cuboid:
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
            return self.n_step_states(new_cubes, n-1)
        
    def n_step_data(self, n=1) -> Xy:
        '''
        Generate data for n moves from the solved state. 
        Remove duplicates and shuffle.
        '''

        # max of n = 4 for now
        assert n <= 4, "n must be less than or equal to 4"

        # find the total number of states
        total_states = 0
        for i in range(n):
            total_states += len(self.actions)**(i+1)

        # initialize the output array
        X = np.zeros((total_states, 288), dtype=bool)
        y = np.zeros((total_states, 1), dtype=float)

        # generate the data for known states
        index = 0
        for i in range(n):
            # assume the it takes i+i moves to get back to the solved state
            state_value = self.gamma**(i+1)

            # get all possible cube configurations for i+1 moves
            cubes = self.n_step_states(Cube(), i+1)
            for cube in cubes:
                # get the state
                state = cube.flat_state()

                # check if the state is already in the array
                if np.where((X == state).all(axis=1))[0].size == 0:
                    X[index] = cube.flat_state()
                    y[index] = state_value
                    index += 1
            
        # remove the unused rows in X and y
        X = X[:index]
        y = y[:index]

        # shuffle the data
        p = np.random.permutation(len(X))
        X, y = X[p], y[p]

        return X, y

    def scrambled_like(self, X:np.ndarray, rat=1.0, moves=20) -> Xy:
        '''
        Generate data for scrambled states. The number of scarambled states is 
        len(X) * rat.
        '''

        # initialize the output array
        total_states = int(len(X) * rat)
        X = np.zeros((total_states, 288), dtype=bool)
        y = np.zeros((total_states, 1), dtype=float)

        # generate random states and assign a value of 0
        for i in range(total_states):
            random_cube = Cube()

            # apply random moves to the cube
            for _ in range(moves):
                action = random.choice(self.actions)
                random_cube(action[0], action[1])
            X[i] = random_cube.flat_state()

        return X, y
    
    def n_greedy_data(self, n:int, data_points=1_000) -> Xy:
        '''scramble a cube n times and solve it with greedy'''

        # initialize data arrays
        X = np.zeros((data_points, len(Cube().flat_state())), dtype=bool)
        y = np.zeros(data_points, dtype=float)

        data_count = 0
        misses = 0
        while data_count < data_points:
            # get fresh cube
            cube = Cube()

            # scramble
            for _ in range(n):
                action = random.choice(self.actions)
                cube(*action)

            # save state
            initial_state = cube.flat_state

            # solve
            c = 0
            solved = False
            while not solved:
                # update move count
                c += 1
                if c > 10:
                    break

                # take greedy move
                action = self.greedy(cube)
                cube(*action)

                # check solved
                solved = cube == Cube()

            # add data point if salved
            if solved:
                X[data_count] = initial_state
                y[data_count] = self.gamma**c
                data_count += 1
            else:
                misses += 1

            # print progress
            print(f'Solved: {data_count}, Unsolved: {misses}', end='\r')

            # if too many cubes are unsolved stop
            if misses > 2*data_points:
                break
        
        return X, y

    def greedy(self, cube:Cube) -> Act:
        '''take the greedy action. return the new cube and action taken'''

        # find possible cubes
        next_cubes = self.n_step_states(cube)

        # find values of states
        best_value = -1
        best_action = 0
        for action, next_cube in enumerate(next_cubes):
            # find the value
            value = self(next_cube)

            # is it best
            if value > best_value:
                best_value = value
                best_action = action

        return self.actions[best_action]