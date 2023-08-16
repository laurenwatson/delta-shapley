import argparse
from data import *
from models import *
from standard_algo.LayeredShapley import LayeredShapley
import torch
import torch.nn.functional as F
import os

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description='Privacy attack parameters')

    # overall experiment parameters
    parser.add_argument("--save_dir", type=str, default='results')

     # data and model parameters
    parser.add_argument("--dataset", type=str, default='cifar10')
    parser.add_argument("--model", type=str, default='resnet18')

    # training parameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    # give index of datapoint to be evaluated
    parser.add_argument("--datapoint", type = int, default = 0)

    return parser


def evaluate_model(model, trainloader, testloader):
    with torch.no_grad():
        losses=[]
        for (images, targets) in trainloader:
            model.zero_grad()
            optimizer.zero_grad()
            if params.device!='cpu':
                images=images.cuda()
                targets=targets.cuda()
            output = model(images)
            loss = F.cross_entropy(output, targets, reduction='sum')
            losses.append(loss.clone().detach())
            output=None
            loss=None
        overall_training_loss=sum(losses)/trainset.__len__()

        losses=[]
        for (images, targets) in testloader:
            model.zero_grad()
            optimizer.zero_grad()
            if params.device!='cpu':
                images=images.cuda()
                targets=targets.cuda()
            output = model(images)
            loss = F.cross_entropy(output, targets, reduction='sum')
            losses.append(loss.clone().detach())
            output=None
            loss=None
        overall_test_loss=sum(losses)/testset.__len__()

    return overall_training_loss, overall_test_loss

if __name__ == '__main__':

    parser = get_parser()
    params = parser.parse_args()
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('running experiment', flush=True)

    if not os.path.exists(params.save_dir):
        os.makedirs(params.save_dir)
        print("New directory created!")


    trainset, testset, num_classes = load_data(params)

    model = return_model(params, num_classes)

    LayeredShapley = LayeredShapley(trainset, testset, L = 1,  beta = 1, c = 1, a = 0.05, b = 0.05, sup = 5, num_classes = num_classes, params = params)
    shapleyvalue = LayeredShapley.run(params.datapoint)
    print("Shapley Value of datapoint: " + str(shapleyvalue))
