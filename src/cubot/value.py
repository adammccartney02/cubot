from torch import nn, optim, tensor, device, cuda, float32
from torch.utils.data import Dataset, DataLoader, random_split
from .cube import Cube
import matplotlib.pyplot as plt
import numpy as np


class CubeDataset(Dataset):

    def __init__(self, cubes:list[Cube], ys:list[float]):
        '''create a dataset of cubes and their values'''
        self.divice = device("cuda" if cuda.is_available() else "cpu")
        self.cubes = cubes
        self.ys = ys
        self.flat_states = []
        self.flat_ys = []

        # create flat states
        for i, cube in enumerate(self.cubes):
            flats = [r_cube.flat_state() for r_cube in cube.roll()]
            self.flat_states.extend(flats)
            self.flat_ys.extend([ys[i] for _ in range(24)])

    def __len__(self):
        '''return the number of samples in the dataset'''
        return len(self.flat_ys)

    def __getitem__(self, idx):
        '''return the flat state and value at the given index'''
        flat = tensor(self.flat_states[idx], dtype=float32).to(self.divice)
        y = tensor(self.flat_ys[idx], dtype=float32).to(self.divice)
        return flat, y
    
    def add(self, cubes:list[Cube], ys:list[float]):
        '''add cubes and ys to the dataset'''
        self.cubes.extend(cubes)
        self.ys.extend(ys)

        # add to flat states
        for i, cube in enumerate(cubes):
            flats = [r_cube.flat_state().astype(float) for r_cube in cube.roll()]
            self.flat_states.extend(flats)
            self.flat_ys.extend([ys[i] for _ in range(24)])

    def export(self, filename):
        '''export the dataset to a file'''
        with open(filename, 'wb') as f:
            np.savez(f, cubes=self.flat_states, ys=self.flat_ys, allow_pickle=True)
        print(f"Dataset exported to {filename}")

class ValueFunction(nn.Module):
    def __init__(self, hidden_shape):
        super(ValueFunction, self).__init__()

        # input and output dimensions
        input_dim = 288 # flat_sate size
        output_dim = 1 # value of the state

        # add hidden layers
        layers = []
        for i, hidden_dim in enumerate(hidden_shape):
            # input dimension is the previous layser's output dimension
            in_dim = input_dim if i == 0 else hidden_shape[i-1]

            # add dense layer and relu activation
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Dropout(0.2))
            layers.append(nn.ReLU())

        # output layer
        layers.append(nn.Linear(hidden_shape[-1], output_dim))

        # create the sequential model
        self.linear_relu_stack = nn.Sequential(*layers)

        # set optimizer and loss function
        self.optimizer = optim.Adam(self.parameters())
        self.loss_fn = nn.L1Loss()

        # move to the device
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)
        self.loss_fn.to(self.device)

        # track losses
        self.losses = []

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
    def __call__(self, cube:Cube):
        '''predict the value of the cube'''

        # put it in evaluation mode
        self.eval()

        # check for a solved cube
        if cube == Cube():
            return 1.0
        else:
            # flatten
            flat = cube.flat_state()

            # to tensor
            tens = tensor(flat, dtype=float32).to(self.device)

            # predict
            return super().__call__(tens).item()
        
    def plot_losses(self):
        '''plot the losses'''

        # get the losses
        train_losses = [loss[0] for loss in self.losses]
        val_losses = [loss[1] for loss in self.losses]

        # plot the losses
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
    
    def train_vf(self, dataset, epochs, batch_size, val_rat=0.2):
        '''train the value function on the dataset'''

        # split dataset into training and validation sets
        train_dataset, val_dataset = random_split(dataset, [1-val_rat, val_rat])
        TrainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        ValLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


        losses = []

        # Training loop
        for epoch in range(epochs):

            # initialize the loss
            train_losses = []
            val_losses = []

            # training mode
            self.train(True)
                
            for X_batch, y_batch in TrainLoader:
                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute the loss
                loss = self.loss_fn(y_pred, y_batch)
                train_losses.append(loss.item())

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

            # validation mode
            self.train(False)

            for X_batch, y_batch in ValLoader:
                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute the loss
                loss = self.loss_fn(y_pred, y_batch)
                val_losses.append(loss.item())

            # display and save losses
            train_loss = sum(train_losses)/len(train_losses)
            val_loss = sum(val_losses)/len(val_losses)
            losses.append((train_loss, val_loss))

            print(' '*100, end='\r')
            print(f"Epoch {epoch+1}/{epochs}, train_loss: {train_loss}, val_loss: {val_loss}", end='\r')
       
        # save losses
        self.losses.extend(losses)
        print()