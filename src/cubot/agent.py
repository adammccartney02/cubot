from .cube import Cube
from .value import ValueFunction
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

class Agent:

    # type alias
    type Xy = tuple[np.ndarray, np.ndarray]
    type Cuboid = list[Cube]|Cube
    type Act = list[int, str]

    def __init__(self, hidden_shape:list[int]):
        # set hyperparameters
        self.hidden_shape = hidden_shape
        self.epsilon = 0.1 # exploration rate
        self.gamma = 0.9 # discount rate

        # set actions
        faces, directions =  np.meshgrid([0, 1, 2, 3, 4, 5], ['cc', 'cw', 'hf'])
        self.actions = list(zip(faces.flatten(), directions.flatten()))

        # build model
        self.model = ValueFunction(hidden_shape=self.hidden_shape)

    def clear_model(self):
        self.model = ValueFunction(hidden_shape=self.hidden_shape)

    def __call__(self, cube:Cube, max_n=20):
        # loop through greedy actions
        c = 0
        solved = False
        while not solved:
            # update move count
            c += 1
            if c > max_n:
                return 0

            # take greedy move
            action = self.greedy(cube)
            cube(*action)

            # check solved
            solved = cube == Cube()

        # return number of moves
        return c
    
    def n_gen(self, n_moves, n_data):
        '''generate n_data points from cubes with n_moves'''
        
        with ProcessPoolExecutor() as executor:
            futers = [executor.submit(self.gen_point, n_moves) for _ in range(n_data)]

            s, u = 0, 0
            data = []
            for f in as_completed(futers):
                # add data
                data.append(f.result()[:])

                # display
                s += f.result()[1] > 0
                u += f.result()[1] == 0
                print(f's: {s}, u: {u}, t: {n_data}', end='\r')
        print()
        X = np.array([X for (X, _) in data])
        y = np.array([[y] for (_, y) in data])
        return X, y

    def gen_point(self, n_moves:int) -> Xy:
        '''generate a data point from a cube with n_moves'''

        # gen cube
        cube = Cube()
        for _ in range(n_moves):
            action = random.choice(self.actions)
            cube(*action)

        # save state
        X = cube.flat_state()
        moves = self(cube, max_n=n_moves+1)
        if moves:
            y = self.gamma**moves
        else: # move == 0 for unsolved cubes
            y = 0

        return X, y

    def train(self, start, stop, step, batch_size=100, cycles=5, epochs=1000):
        # n moves
        X_known, y_known = self.n_step_data(start)

        # scrambled data
        X_scram, y_scram = self.scrambled_like(X_known)

        # combine and shuffle
        X, y = np.vstack((X_known, X_scram)), np.vstack((y_known, y_scram))
        p = np.random.permutation(len(X))
        X, y = X[p], y[p]

        # first training
        print("Dataset size: ", len(y))
        self.model.train_vf(X, y, epochs=epochs)

        for n_moves in range(start, stop, step):
            print("Nuber of Moves: ", n_moves)
            for _ in range(cycles):
                # run greedy
                Xg, yg = self.n_gen(n_moves+1, batch_size)
                solved = [y[0]!=0 for y in yg]
                Xg, yg = Xg[solved], yg[solved]

                # combine and shuffle
                X, y = np.vstack((X, Xg)), np.vstack((y, yg))
                p = np.random.permutation(len(X))
                X, y = X[p], y[p]

                # clear and retrain
                self.clear_model()
                print("Dataset size: ", len(y))
                self.model.train_vf(X, y, epochs=epochs)

        # run greedy
        Xg, yg = self.n_gen(stop, batch_size)
        solved = [y[0]!=0 for y in yg]
        Xg, yg = Xg[solved], yg[solved]

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
    
    def n_scrambled_data(self, n=18, data_points=100):
        '''
        generates data with n moves and assigns a value of gamma**n
        '''

        # initialize
        X = np.zeros((data_points, 288), dtype=bool)
        y = np.ones((data_points, 1))*self.gamma**n

        for i in range(data_points):
            # create a cube
            random_cube = Cube()

            # shuffle the cube
            for _ in range(n):
                action = random.choice(self.actions)
                random_cube(*action)

            # assing to x
            X[i] = random_cube.flat_state()

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
            value = self.model(next_cube)

            # is it best
            if value > best_value:
                best_value = value
                best_action = action

        return self.actions[best_action]