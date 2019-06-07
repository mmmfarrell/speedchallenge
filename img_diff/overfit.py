import numpy
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from img_diff_dataset import ImgDiffDataset
from img_diff_net import ImgDiffNet

import visdom

data_dir = "~/jobs/comma/speedchallenge/data/train"
outputs_file = "~/jobs/comma/speedchallenge/data/train.txt"
dataset = ImgDiffDataset(data_dir, outputs_file)

dataloader = DataLoader(dataset, batch_size=4, num_workers=4)
net = ImgDiffNet()

device = torch.device("cuda:0")
net = net.to(device)

criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

vis = visdom.Visdom()
loss_window = vis.line(X=numpy.zeros((1 ,)),
                       Y=numpy.zeros((1)),
                       opts=dict(xlabel='epoch',
                                 ylabel='Loss',
                                 title='Loss',
                                 ))

running_loss = 0.
for i_batch, sample_batched in enumerate(dataloader):
    # print(i_batch, sample_batched)

    # print(sample_batched['image'].shape)
    inputs = sample_batched['image'].to(device)
    labels = sample_batched['speed'].to(device)
    labels = labels.view(-1, 1)

    # ret = net(sample_batched['image'])

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    # pring every n mini-batches
    if i_batch % 20 == 19:
        print('Iter: {} Loss: {}'.format(i_batch, running_loss / 20))
        epoch_loss = running_loss / 20
        vis.line(X=torch.ones((1,1)).cpu()*i_batch,Y=torch.Tensor([epoch_loss]).unsqueeze(0).cpu(),win=loss_window,update='append')


        running_loss = 0.

    # print(ret.shape)
    # break

