from resnet_mods import PreActResNet18
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = 'cuda:0'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform = transform_train)
trainloader = torch.utils.data.dataloader.DataLoader(trainset, 128, shuffle = True, num_workers = 2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform = transform_test)
testloader = torch.utils.data.dataloader.DataLoader(testset, 1000, shuffle = False, num_workers = 2)

variants = [False, True]

epochs = 90

criterion = nn.CrossEntropyLoss()

num_trials = 10

regnetv2_test_acc = np.zeros((len(variants), num_trials))
regnetv2_loss = np.zeros((len(variants), num_trials, epochs * 391))
regnetv2_train_acc = np.zeros((len(variants), num_trials, epochs))

lrs = [0.2, 0.1, 0.05, 0.02]

for lr in lrs:
    for j in range(num_trials):

        net = PreActResNet18(res = False , num_classes = 100).to(device)
        torch.save(net.state_dict(), 'params_100_{}.pt'.format(j))

        for v in range(len(variants)):
        
            net = PreActResNet18(res = variants[v] , num_classes = 100).to(device)
            net.load_state_dict(torch.load('params_100_{}.pt'.format(j)))
            optimizer = optim.SGD(net.parameters(), lr = 0.1)
            # torch.save(net.state_dict(), 'regnet50_init_{}.pt'.format(j))
            # net.load_state_dict(torch.load('./regnet50_init_{}.pt'.format(j)))
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.2)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            k = 0
            for i in range(epochs):
                print('Epoch: {}'.format(i))
                correct = 0
                total = 0
                for data in trainloader:
                    optimizer.zero_grad()

                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = net(inputs)

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = criterion(outputs, labels)
                    regnetv2_loss[v, j, k] = loss
                    loss.backward()
                    optimizer.step()
                    k = k + 1
                scheduler.step()

                regnetv2_train_acc[v, j, i] = correct/total
                print('Training accuracy: {}'.format(correct / total))

            # torch.save(net.state_dict(), 'regnet152_trained_{}.pt'.format(j))
            # net.load_state_dict(torch.load('resnet18.pt'))

            with torch.no_grad():
                correct = 0
                total = 0
                for data in testloader:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = net(inputs)
                    # print(outputs.shape)

                    _, predicted = torch.max(outputs, 1)
                    # print(predicted.shape)
                    # print(labels.shape)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print('Test accuracy: {}'.format(correct / total))

                regnetv2_test_acc[v, j] = correct / total

    np.save(f'comparison_100_test_{lr:g}.npy', regnetv2_test_acc)
    np.save(f'comparison_100_loss_{lr:g}.npy', regnetv2_loss)
    np.save(f'comparison_100_train_{lr:g}.npy', regnetv2_train_acc)