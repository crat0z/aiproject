import torch
import torch.nn as nn

import os

# game is 15 x 15 grid
input_size = 15 * 15
# this will probably need changes
hidden_size = input_size * input_size
# whether to go up/down/left/right. this might need to also be changed
# to forward, left, right, since going "backwards" is not a valid move ever
output_size = 3


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # self.layers = nn.Sequential(
        #    nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, input_size), nn.ReLU(), nn.Linear(input_size, output_size))

        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        # x = torch.reshape(x, (1, 1, 15, 15)).to(device="cpu")

        return self.layers(x)
