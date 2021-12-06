import torch
import torch.nn as nn

import os

# game is 16 x 16 grid
input_size = 16 * 16
# this will probably need changes
hidden_size = input_size * input_size
# whether to go up/down/left/right. this might need to also be changed
# to forward, left, right, since going "backwards" is not a valid move ever
output_size = 3


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, input_size), nn.ReLU(), nn.Linear(input_size, output_size))

    def forward(self, x):
        return self.layers(x)

    def save(self, file_name="model"):
        folder = "./model"

        if not os.path.exists(folder):
            os.makedirs(folder)

        file = os.path.join(folder, file_name)
        torch.save(self.state_dict(), file)
        # might be worth it to save optimizer here too
