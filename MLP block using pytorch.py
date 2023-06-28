import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MLPBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))
        self.activations = nn.ModuleList()
        for i in range(num_layers - 1):
            self.activations.append(nn.ReLU())
        self.activations.append(nn.Identity())

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activations[i](x)
        return x
