import torch
import math
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms

def load_data(params, binary_trainsize=2000, binary_testsize=1000):

    """
    Returns data with first index the datapoint, each datapoint has features then label.
    """

    dataset_name=params.dataset
    
    if dataset_name=='mnist':
        
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        
        dataset = torchvision.datasets.MNIST(root='../mnist', train=True, download=False, transform=transform)

        print('Loaded MNIST data, datasize:', len(dataset))
        train_size=math.floor(len(dataset)*0.8)

        # random.shuffle(dataset)
        data=[]
        for i in np.arange(len(dataset)):
            data.append([dataset[i][0], dataset[i][1]])
        
        return data[:train_size], data[train_size:], 10
    
    elif dataset_name=='fashionmnist':
        
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        
        dataset = torchvision.datasets.FashionMNIST(root='../../fashionmnist', train=True, download=False, transform=transform)

        print('Loaded FashionMNIST data, datasize:', len(dataset))
        train_size=math.floor(len(dataset)*0.8)

        # random.shuffle(dataset)
        data=[]
        for i in np.arange(len(dataset)):
            data.append([dataset[i][0], dataset[i][1]])
        
        return data[:train_size], data[train_size:], 10
    
    elif dataset_name=='binary-fashionmnist':
        
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])
        
        dataset = torchvision.datasets.FashionMNIST(root='../fashionmnist', train=True, download=False, transform=transform)

        train_data, test_data = [],[]
        count_0,count_6 = 0,0
        for i in np.arange(len(dataset)):
            c=dataset[i][1]
            if c==0 and count_0<binary_trainsize/2:
                train_data.append([dataset[i][0], 0])
                count_0+=1
            elif c==0 and count_0>=binary_trainsize/2 and count_0<(binary_trainsize+binary_testsize)/2:
                test_data.append([dataset[i][0], 0])
                count_0+=1
            elif c==6 and count_6<binary_trainsize/2:
                train_data.append([dataset[i][0],1])
                count_6+=1
            elif c==6 and count_6>=binary_trainsize/2 and count_6<(binary_trainsize+binary_testsize)/2:
                test_data.append([dataset[i][0], 1])
                count_6+=1
        
        print('Loaded Binary FashionMNIST data, datasize:', len(train_data), 'test size ', len(test_data))

        
        return train_data, test_data, 2

    elif dataset_name=='cifar10':

        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        trainset = torchvision.datasets.CIFAR10(root='../cifar10', train=True,download=False, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='../cifar10', train=False,download=False, transform=transform)


        return trainset, testset, 10

    elif dataset_name=='cifar100':

        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        trainset = torchvision.datasets.CIFAR100(root='../cifar100', train=True,download=False, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='../cifar100', train=False,download=False, transform=transform)

        return trainset, testset, 100
