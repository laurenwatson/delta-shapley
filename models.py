import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression

class CNN(torch.nn.Module):
    def __init__(self, num_classes=10 ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,  num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SmallCNN(torch.nn.Module):
    def __init__(self, num_classes=2 ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(864, 84)
        self.fc2 = nn.Linear(84,  num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def return_model(params, num_classes, seed=2):
    if params.model=='cnn':
        if seed:
            torch.manual_seed(seed)
        model=CNN(num_classes)
    elif params.model=='small-cnn':
        if seed:
            torch.manual_seed(seed)
        model=SmallCNN(num_classes)
        if params.pretrain:
            print('using pretrained model')
            model_filename='fashionmnist_smallcnn_pretrained_model'
            with open(model_filename, 'rb') as f:
                model_state_dict=torch.load(f)
            model.load_state_dict(model_state_dict)
    elif params.model=='logistic':
        solver = 'liblinear'
        max_iter =  5000
        lam=0.1
        model = LogisticRegression(solver=solver, max_iter=max_iter, random_state=seed, C=1/lam)

    if params.model!='logistic':
        if torch.cuda.is_available():
            #print('Using Cuda')
            model=model.cuda()

    return model
