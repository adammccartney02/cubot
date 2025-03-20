from torch import nn

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
            layers.append(nn.ReLU())

        # output layer
        layers.append(nn.Linear(hidden_shape[-1], output_dim))

        # create the sequential model
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits