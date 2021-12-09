import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            # deepmind paper uses in_channel = 4
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            # papers don't make this 3136 explicit, so i found this by simply plugging in
            # (1,1,86,86) tensor into above and seeing pytorch error
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        return self.layers(x)
