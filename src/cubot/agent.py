from .cube import Cube
from .value import ValueFunction
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

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

    def __call__(self, in_cube:Cube, max_n=20):
        # copy the cube
        cube = in_cube.copy()

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
    
    def n_gen(self, n:int) -> tuple[list[Cube], list[float]]:
        '''generate n data points distibuted evenly across 1 to 18 moves'''

        # cube_list will be rolled before training
        cube_list = []
        value_list = []

        #################### 1 move ####################

        # find all move on face 0
        cube1 = Cube()
        for _ in range(3):
            cube1(0, 'cc')
            cube_list.append(cube1.copy())
            value_list.append(self.gamma)

        #################### 2 moves ####################

        # make a copy of all 1 move cubes (white)
        cube_list_1move = deepcopy(cube_list)

        # find adjecent move configurations (red)
        for cube in cube_list_1move:
            for _ in range(3):
                cube(1, 'cc')
                cube_list.append(cube)
                value_list.append(self.gamma**2)

        # find opposite move configuration (yellow)
        for cube in cube_list_1move:
            for _ in range (3):
                cube(5, 'cc')
                cube_list.append(cube)
                value_list.append(self.gamma**2)

        #################### n moves ####################

        # find the value of n random moves
        def val(gamma:float, n:int) -> float:
            n = min(20, n)
            value = gamma ** n*(1-n*0.025)
            return value

        # fill the cube list with partially shuffled cubes
        for i in range(int(len(cube_list), n)):
            # get fresh cube
            cube = Cube()
            moves = random.randint(3, 18)

            # scramble
            for _ in range(moves):
                action = random.choice(self.actions)
                cube(*action)

            # assign X and y
            cube_list.append(cube)
            value_list.append(val(moves))

        return cube_list, value_list

    def point(self, cube:Cube, i:int):

        # get a random orientation of the cube
        rand_cube = np.random.choice(cube.roll())

        # solve the cube
        moves = self(rand_cube)

        # assign value if solved
        if moves:
            s = True
            y = self.gamma ** moves
        else:
            s = False
            y = -100

        return s, y, i

    def train(self, dataset_size=1000, epochs=1600, max_i=10):

        # generate cubes
        X_cubes, y_cubes = self.n_gen(dataset_size)
        np.savez('initial_cubes.npz', X_cubes=X_cubes, y_cubes=y_cubes)

        # initialize an array to track if cubes have been solved
        s_cubes = np.zeros_like(y, dtype=bool)

        # generate states for rolled cubes
        X_flat, y_flat = [], []
        for i, cube in enumerate(X_cubes):
            roll = cube.roll()
            for r_cube in roll:
                X_flat.append(r_cube.flat_state())
                y_flat.append(y_cubes[i])

        # shuffle
        p = np.random.permutation(X_flat)
        X_flat, y_flat = X_flat[p], y_flat[p]

        # loop until good at rubix cube
        acc = 0
        while acc < 0.9:
            # train value function
            self.model.train_vf(X_flat, y_flat, epochs=epochs)

            # use 12 cores to solve all cubes in X_cubes
            with ProcessPoolExecutor(max_workers=12) as executor:
                # submit 
                futers = [executor.submit(self.point, cube, i) for i, cube in enumerate(X_cubes)]

                # extract data
                for f in as_completed(futers):
                    s, y, i = f.result()

                    # updata y and s
                    y_cubes[i] = y
                    s_cubes[i] = s

                    # display
                    solved += s == 1
                    unsolved += s == 0
                    print(f's: {solved}, u: {unsolved}, t: {len(y_cubes)}', end='\r')
                print()

            # find accuracy
            acc = sum(s_cubes)/len(s_cubes)

            # stop at max iterations
            c += 1
            if c > max_i:
                break

        
    def greedy(self, cube:Cube) -> Act:
        '''take the greedy action. return the new cube and action taken'''
        
        # init
        best_value = -100
        best_action_idx = 0

        # loop through all actions
        for i, action in enumerate(self.actions):

            # get a posible next cube
            next_cube = cube.copy()
            next_cube(*action)

            # evaluate the cube
            value = self.model(next_cube)

            # find best value and action
            if value > best_value:
                best_value = value
                best_action_idx = i

        # return the best action
        return self.actions[best_action_idx]