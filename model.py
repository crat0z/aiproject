import torch
import torch.nn as nn

# Conv2d "Applies a 2D convolution over an input signal composed of several input planes."
# Conv2d are a little picky about input tensors, given a tensor with shape (A, B, C, D):
#   - tensor must be 4d
#   - B must match in_channels
#   - A can be arbitrary, it is the "batch_size", since typically CNNs are used in computer vision
#     applications, processing many images at once is much faster this way
#   - C,D >= kernel_size
#
# The output of a Conv2d layer can be easily calculated, for example with an input vector
# of (N, C, X, Y):
#   - N is always the same.
#   - C is always the out_channels
#   - X = 1 + floor((X - kernel_size)/stride)
#   - Y = 1 + floor((Y - kernel_size)/stride)

# e.g. nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=2)
# with input (4, 2, 5, 5) -> (4, 8, 2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            # deepmind paper uses in_channel = 4
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
            # (N, 32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=4, stride=2),
            # (N, 64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1),
            # (N, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
            # (N, 64 * 7 * 7 = 3136)
            nn.Linear(3136, 512),
            # (N, 512)
            nn.ReLU(),
            nn.Linear(512, 3)
            # (N, 3)
        )

    def forward(self, x):

        return self.layers(x)
