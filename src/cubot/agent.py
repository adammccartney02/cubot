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
    
    def n_gen(self, n:int) -> list[Cube]:
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

        # find the value of a random move
        # 15/18 are productive
        # 1/18 undoes the previous move
        #     1/6 chance that 2 moves ago the opposite face was moved
        #     2/18 moves are unproductive
        #     1/18 moves undoes the previous move
        # random_move_factor = 0.74 # 14/18 - (4/18)/6

        # n = 1 -> 1, max at n = 20,
        def val(n:int) -> float:
            n = min(20, n)
            return n * (40/39 - n/39)

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
            value_list.append(self.gamma ** val(moves))

        return cube_list, value_list

        


    def train(self, start, stop, step, batch_size=10, cycles=5, epochs=3200):
        # n moves
        X0, y0 = self.n_step_data(start)
        good_Xs = [X0]
        good_ys = [y0]
        print("Grid data size: ", len(y0))

        # scrambled data
        ok_Xs = []
        ok_ys = []
        for i in range(start+step*2, stop, step):
            Xs, ys = self.n_scrambled_data(i, batch_size)
            ok_Xs.append(Xs)
            ok_ys.append(ys)
        print("Initial random data size: ", batch_size*len(ok_ys))


        # combine and shuffle
        X, y = np.vstack((*ok_Xs, *good_Xs)), np.vstack((*ok_ys, *good_ys))
        p = np.random.permutation(len(X))
        X, y = X[p], y[p]

        # first training
        self.model.train_vf(X, y, epochs=epochs)

        # intrement number of moves
        for n_moves in range(start+step, stop, step):
            print('*'*50)
            print("Number of Moves: ", n_moves)
            print()

            # cycle at n moves
            for _ in range(cycles):
                # run greedy
                Xg, yg = self.n_gen(n_moves, batch_size)
                solved = [y[0]!=0 for y in yg]
                Xg, yg = Xg[solved], yg[solved]
                print("Data added: ", len(yg))

                # add new data to the good data
                good_Xs.append(Xg)
                good_ys.append(yg)

                # combine and shuffle
                X, y = np.vstack((*ok_Xs, *good_Xs)), np.vstack((*ok_ys, *good_ys))
                p = np.random.permutation(len(X))
                X, y = X[p], y[p]

                # clear and retrain
                self.model = ValueFunction(hidden_shape=self.hidden_shape)
                print("Dataset size: ", len(y))
                self.model.train_vf(X, y, epochs=epochs)
                print()

            # remove ok data at lowest number of moves
            if len(ok_ys) > 0:
                print("Data removed: ", len(ok_ys[0]))
                ok_Xs = ok_Xs[1:]
                ok_ys = ok_ys[1:]

        # run greedy
        Xg, yg = self.n_gen(stop, batch_size)
        
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