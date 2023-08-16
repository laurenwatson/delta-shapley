import argparse
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data import *
from models import *
from passwise_sampling_sc.LShapley_Sample2_sc import LayeredShapley_Sample2
import torch
import os


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
    parser.add_argument("--prob_type", type=str, default='select-layers-middle-third')
    parser.add_argument("--datasize", type=int, default=100)


    # training parameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    # give index of datapoint to be evaluated
    parser.add_argument("--eval_datapoint_ids", type = list, default = [0,1,2])


    return parser


if __name__ == '__main__':

    parser = get_parser()
    params = parser.parse_args()
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running experiment', flush=True)

    if not os.path.exists(params.save_dir):
        os.makedirs(params.save_dir)
        print("New directory created!")

    num_classes=2
    LayeredShapley = LayeredShapley_Sample2( num_classes = num_classes, params = params)
    shapleyvalue = LayeredShapley.run(params.eval_datapoint_ids, params)
    print("Shapley Value of datapoint: " + str(shapleyvalue))
