# cube

In this repository you will find a vertual rubik's cube, and a reinforcement learning agent that learns to solve the cube.
The agrent uses a greedy policy to move the cude closer to being solved. The value of each state of the cube is approximated using a neural network. Durring the training phase the argent is shown cubes with n number of random moves. If the Agent successfully solves the cube, the original state of the cube is added to a training set of known state-value data points that is used for training the value function model. The cubes get more mixed up as training progresses. At any state a rubic's cube is only 20 moves from solved. So, once 20 moves is reached the agent should be able to solve any cube. 

warning: This model was built to run specifically on my machine. It uses 12 cores and a GPU. If you do not have that, it will break :(