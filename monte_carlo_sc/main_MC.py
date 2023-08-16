import argparse
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data_utils import *
from models import *
from monte_carlo_sc.MonteCarloShapley import MonteCarloShapley
import os
import pandas as pd


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

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
    parser.add_argument("--dataset", type=str, default='adult')
    parser.add_argument("--model", type=str, default='logistic')
    parser.add_argument("--pretrain", type=bool_flag, default=False)
    parser.add_argument("--datasize", type=int, default=100)


    # training parameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    # give index of datapoint to be evaluated
    parser.add_argument("--datapoint", type = int, default = 0)
    parser.add_argument("--eval_datapoint_ids", type = list, default = [i for i in range(10)])


    return parser


if __name__ == '__main__':

    parser = get_parser()
    params = parser.parse_args()

    print('running experiment', flush=True)
    if not os.path.exists(params.save_dir):
        os.makedirs(params.save_dir)
        print("New directory created!")

    num_classes=2
    X,y=load_data(params.dataset)
    train_size=int(len(X)*0.8)
    X_train=X[:train_size]
    y_train=y[:train_size]
    X_test=X[train_size:]
    y_test=y[train_size:]
    # trainset, testset, num_classes = load_data(params, binary_trainsize=params.datasize, binary_testsize=params.datasize)
    print("loaded trainset")
    MC = MonteCarloShapley(X_train, y_train, X_test, y_test, L = 1,  beta = 1, c = 1, a = 0.05, b = 0.05, sup = 5, num_classes = num_classes, params = params)
    print("starting run")
    shapleyvalues = MC.run(params.eval_datapoint_ids, params)


    print("Shapley Value of datapoints: " + str(params.eval_datapoint_ids)+" is "+ str(shapleyvalues))
