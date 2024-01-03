import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

trainset = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
indices = torch.randperm(60000)[:32]
trainset = torch.utils.data.Subset(trainset, indices)
trainloader = torch.utils.data.dataloader.DataLoader(trainset, 32, shuffle = False, num_workers = 2)

class Block1(nn.Module):
    def __init__(self, res = False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(5)
        self.relu1 = nn.ReLU()
        self.lin1 = nn.Conv2d(5, 5, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(5)
        self.relu2 = nn.ReLU()
        self.lin2 = nn.Conv2d(5, 5, 5, 1, 2)
        self.res = res

    def forward(self, x):
        branch = self.lin2(self.relu2(self.bn2(self.lin1(self.relu1(self.bn1(x))))))
        if self.res == True:
            return x + branch
        else:
            return branch

class Block2(nn.Module):
    def __init__(self, res = False, projection = True):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(5)
        self.relu1 = nn.ReLU()
        self.cproj = nn.Conv2d(5, 20, 1, 3)
        self.lin1 = nn.Conv2d(5, 20, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(20)
        self.relu2 = nn.ReLU()
        self.lin2 = nn.Conv2d(20, 20, 3, 1)
        self.res = res
        self.projection = projection
        self.avg = nn.AvgPool2d(3, 3)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        branch = self.lin2(self.relu2(self.bn2(self.lin1(out))))
        if self.res == True:
            if self.projection == True:
                shortcut = self.cproj(out)
                return (shortcut + branch).flatten(1)
            else:
                shortcut = F.pad(3*self.avg(out), (0, 0, 0, 0, 7, 8), 'constant', 0.) + self.cproj(out)
                return (shortcut + branch).flatten(1)
        else:
            return branch.flatten(1)


class BNNet(nn.Module):
    def __init__(self, res = False, proj = False):
        super().__init__()
        self.conv = nn.Conv2d(1, 5, 5, 1)
        self.avg = nn.AvgPool2d(2, 2)
        self.Block1 = Block1(res)
        self.Block2 = Block2(res, proj)
        self.bn = nn.BatchNorm1d(320)
        self.relu = nn.ReLU()
        self.lin = nn.Linear(320, 10)

    def forward(self, x):
        out = self.avg(self.conv(x))
        out = self.Block1(out)
        out = self.Block2(out)
        out = self.relu(self.bn(out))
        out = self.lin(out)
        return out


device = 'cuda:0'

epochs = 30


res = [[False, False], [True, True], [True, False]]
res_indices = [0, 1, 2]

criterion = nn.CrossEntropyLoss()
num_trials = 10

svals = np.zeros((num_trials, 3, 50, 320))
lossvals = np.zeros((num_trials, 3, 50))

for trials in range(num_trials):

    net = BNNet().to(device)
    torch.save(net.state_dict(), 'params.pt')

    for idx in res_indices:

        net = BNNet(res[idx][0], res[idx][1]).to(device)
        net.load_state_dict(torch.load('params.pt'))
        optimiser = torch.optim.SGD(net.parameters(), lr = 0.1)

        for i in range(epochs):
            for data, labels in trainloader:

                data = data.to(device)
                labels = labels.to(device)

                optimiser.zero_grad()

                inputs = net.avg(net.conv(data)).requires_grad_(True)

                jac2 = torch.autograd.functional.jacobian(nn.Sequential(net.Block1, net.Block2, net.bn, net.relu, net.lin), inputs)
                jac2 = jac2.flatten(0, 1)
                jac2 = jac2.flatten(1)
                svals[trials, idx, i] = torch.linalg.svdvals(jac2).cpu()

                data.requires_grad_(False)

                outputs = net(data)
                loss = criterion(outputs, labels)
                print(loss)
                lossvals[trials, idx, i] = loss
                loss.backward()

                optimiser.step()

np.save('svalsconv_trials.npy', svals)
np.save('lossvalsconv_trials.npy', lossvals)


