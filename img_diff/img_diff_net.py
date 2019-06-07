from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

class ImgDiffNet(nn.Module):
    # Input size is
    def __init__(self):
        super(ImgDiffNet, self).__init__()
        # This is channels in, channels out, kernel size
        self.num_chan1 = 5
        self.conv1 = nn.Conv2d(1, self.num_chan1, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.num_chan2 = 10
        self.conv2 = nn.Conv2d(5, self.num_chan2, 3, padding=1)

        self.num_fc1 = 64
        self.fc1 = nn.Linear(self.num_chan2 * 15 * 20, self.num_fc1)

        self.num_fc2 = 1
        self.fc2 = nn.Linear(self.num_fc1, self.num_fc2)

    def forward(self, x):
        # Input size is (batch, 1, 60, 80)
        x = F.relu(self.conv1(x))

        # (batch, self.num_chan1, 60, 80)
        x = self.pool(x)

        # (batch, self.num_chan1, 30, 40)
        x = self.pool(F.relu(self.conv2(x)))

        # (batch, self.num_chan2, 15, 20)
        x = x.view(-1, self.num_chan2 * 15 * 20)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return (x)
