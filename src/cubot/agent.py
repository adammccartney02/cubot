from .cube import Cube
from .value import ValueFunction, CubeDataset
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

    def lookahead(self, cube:Cube) -> Act:

        # init
        best_value = -100
        best_action_idx = 0

        # loop through all actions
        for i, action in enumerate(self.actions):

            # get a posible next cube
            next_cube = cube.copy()
            next_cube(*action)

            # move greedy
            next_action = self.greedy(next_cube)
            next_cube(*next_action)
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
        cubes = []

        # loop until solved
        while cube != Cube():
            # update cubes
            cubes.append(cube.copy())

            # get best action
            action = self.greedy(cube)
            cube(*action)

            # break if it takes too long
            if len(cubes) > 20:
                return cube, -1

        # set values of the cubes
        n_moves = len(cubes)
        ys = [[self.gamma**(n_moves-i)] for i in range(len(cubes))]
        np.savez('cube_data.npz', cubes=cubes, ys=ys, allow_pickle=True)

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
            value_list.append([self.gamma])

        #################### 2 moves ####################

        # make a copy of all 1 move cubes (white)
        cube_list_1move = deepcopy(cube_list)

        # find adjecent move configurations (red)
        for cube in cube_list_1move:
            for _ in range(3):
                cube(1, 'cc')
                cube_list.append(cube)
                value_list.append([self.gamma**2])

        # find opposite move configuration (yellow)
        for cube in cube_list_1move:
            for _ in range (3):
                cube(5, 'cc')
                cube_list.append(cube)
                value_list.append([self.gamma**2])
        
        # return the cubes and values
        return cube_list, value_list

    def train(self, n_new_cubes=10, n_zeros=10, 
              acc_min=0.95, max_cycles=10):

        # initialize dataset
        X_cubes, y_cubes = self.initial_gen()

        # generate zeros
        X_zeros = [Cube() for _ in range(n_zeros)]
        y_zeros = [[self.gamma**18] for _ in range(n_zeros)]
        for i in range(n_zeros):
            # generate a random cube
            cube = Cube()

            # shuffle the cube. 20 moves in gods number, but we want to be sure
            for _ in range(30):
                action = random.choice(self.actions)
                cube(*action)

            # add the cube to the dataset
            X_zeros[i] = cube.copy()

        # create the dataset
        dataset = CubeDataset(X_cubes, y_cubes)
        dataset.add(X_zeros, y_zeros)

        # dataset.export('data2.npz')
        # return

        # step through moves from 3 to 20
        for n in range(3, 21):
            # print header
            print('*'*50)
            print('Moves: ', n)

            # Cycle
            cycle_count = 0
            acc = 0.0
            while acc < acc_min:

                # train value function
                print('Training Data Size: ', len(dataset))
                self.model.train_vf(dataset, int(len(dataset)/10), 240)
                self.model.plot_losses()

                # # does the model just pick an avarage value for all cubes?
                # test_cube = Cube()
                # test_cube(0, 'cc')
                # a1 = self.model(test_cube)
                # test_cube(1, 'cw')
                # a2 = self.model(test_cube)
                # if (a1 - a2) < 0.05:
                #     print('Model is not learning')
                #     return X_cubes, y_cubes

                # create more data
                X_new = np.array([Cube() for _ in range(n_new_cubes)])
                for i in range(n_new_cubes):
                    # generate a random cube
                    cube = Cube()
                    for _ in range(n-1):
                        action = random.choice(self.actions)
                        cube(*action)

                    # add the cube to the dataset
                    X_new[i] = cube.copy()

                # solve
                X_cubes = []
                y_cubes = []
                with ProcessPoolExecutor(max_workers=12) as executor:
                    # submit
                    futers = [executor.submit(self.MC, cube) for cube in X_new]

                    # extract data
                    solved, unsolved = 0, 0
                    for f in as_completed(futers):
                        # get the result
                        cubes, ys = f.result()

                        # updata y and s
                        if ys != -1:
                            # add cubes and values to the dataset
                            X_cubes += cubes
                            y_cubes += ys

                            # update number of solved cubes
                            solved += 1
                        else:
                            # update number of unsolved cubes
                            unsolved += 1

                        # display
                        print(f's: {solved}, u: {unsolved}, t: {n_new_cubes}', end='\r')
                    print()

                # evaluate the model
                acc = solved / (solved + unsolved)
                cycle_count += 1
                print(f'Accuracy ({cycle_count}): {acc}')

                # add the new cubes to the dataset
                dataset.add(X_cubes, y_cubes)

                # check for max cycles
                if cycle_count > max_cycles:
                    print('Max cycles reached')
                    break

        return X_cubes, y_cubes