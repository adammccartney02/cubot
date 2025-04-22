from torch import nn, optim, tensor, device, cuda, float32, from_numpy
from torch.utils.data import Dataset
from .cube import Cube


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
            self.flat_ys.extend([self.ys[i]]*24)

    def __len__(self):
        '''return the number of samples in the dataset'''
        return len(self.flat_ys)

    def __getitem__(self, idx):
        '''return the flat state and value at the given index'''
        flat = tensor(self.flat_states[idx]).to(self.divice)
        y = tensor(self.flat_ys[idx]).to(self.divice)
        return flat, y
    
    def add(self, cubes:list[Cube], ys:list[float]):
        '''add cubes and ys to the dataset'''
        self.cubes.extend(cubes)
        self.ys.extend(ys)

        # add to flat states
        for i, cube in enumerate(cubes):
            flats = [r_cube.flat_state() for r_cube in cube.roll()]
            self.flat_states.extend(flats)
            self.flat_ys.extend([ys[i]]*24)

class ValueFunction(nn.Module):
    def __init__(self, hidden_shape):
        super(ValueFunction, self).__init__()

        # input and output dimensions
        input_dim = 288 # flat_sate size
        output_dim = 1 # value of the state

        # add hidden layers
        layers = []
        for i, hidden_dim in enumerate(hidden_shape):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_shape[i-1], hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())

        # output layer
        layers.append(nn.Linear(hidden_shape[-1], output_dim))

        # create the sequential model
        self.linear_relu_stack = nn.Sequential(*layers)

        # set optimizer and loss function
        self.optimizer = optim.Adam(self.parameters())
        self.loss_fn = nn.MSELoss()

        # add to the device
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)
        self.loss_fn.to(self.device)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
    def __call__(self, cube:Cube):

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

    
    def train_vf(self, X, y, epochs=5000):

        self.train()

        # Convert numpy arrays to torch tensors
        X_tensor = tensor(X, dtype=float32).to(self.device)
        y_tensor = tensor(y, dtype=float32).to(self.device)

        losses = []

        # Training loop
        for epoch in range(epochs):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self.forward(X_tensor)

            # Compute the loss
            loss = self.loss_fn(y_pred, y_tensor)
            losses.append(loss.item())

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            if (epoch+1) % 100 == 0:
                print(' '*100, end='\r')
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}", end='\r')
        print()