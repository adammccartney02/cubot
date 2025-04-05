from .cube import Cube
from .value import ValueFunction
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import time
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

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
    
    def MC(self, cube:Cube) -> tuple[Cuboid, list[float]]:
        '''every state Monte Carlo'''

        # init
        cubes = [cube.copy()]

        # loop until solved
        while cube != Cube():
            # get best action
            action = self.greedy(cube)
            cube(*action)

            # update cubes
            cubes.append(cube.copy())

            # break if it takes too long
            if len(cubes) > 20:
                return cube, -100

        # set values of the cubes
        n_moves = len(cubes)
        ys = [self.gamma**(n_moves-i) for i in range(len(cubes))]

        return cubes, ys

    def initial_gen(self) -> tuple[list[Cube], list[float]]:
        '''generate cubes for with 1 or 2 moves'''

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
        
        # return the cubes and values
        return cube_list, value_list

    def monte_carlo(self, cube:Cube, i:int):

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

    def train(self, dataset_size=10000, epochs=1600, max_cycles=10, load_data=False):

        if not load_data:
            # generate cubes
            X_cubes, y_cubes = self.n_gen(dataset_size)

            # initialize an array to track if cubes have been solved
            s_cubes = np.zeros_like(y_cubes, dtype=bool)

            # generate states for rolled cubes
            X_flat = np.zeros((24*dataset_size, 288), dtype=bool)
            y_flat = np.zeros((24*dataset_size, 1), dtype=float)
            s_flat = np.zeros((24*dataset_size, 1), dtype=bool)

            flat_idx = 0
            for cube_idx, cube in enumerate(X_cubes):
                # generate 24 cubes
                roll = cube.roll()
                for r_cube in roll:
                    # add the flat state
                    X_flat[flat_idx] = r_cube.flat_state()
                    y_flat[flat_idx] = y_cubes[cube_idx]
                    s_flat[flat_idx] = s_cubes[cube_idx]
                    flat_idx += 1
                print(f'Generating flat states: {cube_idx+1}/{dataset_size}', end='\r')
            print()

            # save
            np.savez('cube_dataset.npz', X_flat, y_flat, s_flat, 
                     X_cubes, y_cubes, s_cubes)
        else: 
            data = np.load('cube_dataset.npz', allow_pickle=True)
            X_flat, y_flat, s_flat = data['arr_0'], data['arr_1'], data['arr_2']
            X_cubes, y_cubes, s_cubes = data['arr_3'], data['arr_4'], data['arr_5']



        # shuffle
        X_flat, y_flat, s_flat = shuffle(X_flat, y_flat, s_flat)
        p = np.random.permutation(len(y_flat))
        X, y, s = X_flat[p], y_flat[p], s_flat[p]

        # loop until good at rubix cube
        acc = 0
        cycle_count = 0
        while acc < 0.9:
            # print header
            print('*'*50)
            print('Cycle: ', cycle_count)

            # train value function
            self.model.train_vf(X, y, epochs=epochs)

            # use 12 cores to solve all cubes in X_cubes
            with ProcessPoolExecutor(max_workers=12) as executor:
                # submit 
                futers = [executor.submit(self.point, cube, i) for i, cube in enumerate(X_cubes)]

                # extract data
                solved, unsolved = 0, 0
                for f in as_completed(futers):
                    s_res, y_res, i_res = f.result()

                    # updata y and s
                    if s_res:
                        y_cubes[i_res] = y_res
                        s_cubes[i_res] = True
                        solved += 1
                    else:
                        unsolved += 1

                    # display
                    print(f's: {solved}, u: {unsolved}, t: {len(y_cubes)}', end='\r')
                print()

            # update y_flat
            j = 0
            for i in range(len(y_cubes)):
                for _ in range(24):
                    y_flat[j] = y_cubes[i]
                    j += 1

            plt.plot(y_flat)
            plt.show
            y = y_flat[p]

            # find accuracy
            acc = sum(s_cubes)/len(s_cubes)
            print('Accuracy: ', acc)

            # stop at max iterations
            cycle_count += 1
            if cycle_count > max_cycles:
                break