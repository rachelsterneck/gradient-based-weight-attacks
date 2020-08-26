"""
Trains a language model from the command line.
"""
import argparse
import random
import os

import torch

from data import load_data
from model import build_model
from attack import attack_layers

# Command line arguments
parser = argparse.ArgumentParser(description="Weight based attacks")

parser.add_argument("--data", type=str, default="MNIST",  
                    help="Dataset: MNIST, CIFAR10, CIFAR100, or ImageNet")
parser.add_argument("--model_name", type=str, default="ResNet18", 
                    help="Network architecture: VGG19, ResNet18, ResNet18-pretrained-imagenet, LeNet")
parser.add_argument("--model_path", type=str, default=None,
                    help="Path to pretrained model file")
parser.add_argument("--data_path", type=str, default="./data",  
                    help="Local path to data")
parser.add_argument('--epsilons', nargs='+', 
                    help='attack epsilons separated by spaces')
parser.add_argument("--num_weights", type=int, default=1,
                    help="Number of weights to attack")
parser.add_argument('--layer_idx', nargs='+', 
                    help='n-1 where n is layer separated by spaces')
parser.add_argument("--batch_size", type=int, default=128,
                    help="Batch size")
parser.add_argument("--num_workers", type=int, default=0,
                    help="The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed")
parser.add_argument("--cuda", action="store_true",
                    help="Set this flag to use CUDA")
# parser.add_argument("--save", type=str, default="output.png",
#                     help="Path to save the final output images")

kwargs = vars(parser.parse_args())

# Set seed
if kwargs["seed"] is not None:
    random.seed(kwargs["seed"])
del kwargs["seed"]

# Load data
print('==> Preparing dataset %s' % kwargs["data"])
device = torch.device("cuda" if kwargs["cuda"] else "cpu")
testloader, num_classes = load_data(kwargs["data"], kwargs["data_path"], kwargs["batch_size"], kwargs["num_workers"], device)

del kwargs["data"]
del kwargs["cuda"]
del kwargs["batch_size"]

# Build the model
print('==> Building model %s' % kwargs["model_name"])
model, layer_idx = build_model(kwargs["model_name"], num_classes, device, kwargs["model_path"], kwargs["layer_idx"])
model.eval()

# Run attack
print('==> Running attack | Num Weights: {} | Epsilons: {} | Layer_idx : {}'.format(kwargs["num_weights"], kwargs["epsilons"], kwargs["layer_idx"]))
accs, confs = attack_layers(model, device, testloader, kwargs["epsilons"], kwargs["num_weights"], layer_idx)


