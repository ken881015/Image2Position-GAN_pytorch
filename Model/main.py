import numpy as np
import torch
import torch.nn as nn
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter

from Generator import *
from Discriminator import *


# parser configuration
parser = argparse.ArgumentParser()

# file io
parser.add_argument("--input_dir", required=True, help="path to folder containing training data")
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--log_file", default="training_log", help="path to log file")

# model structure and parameter
parser.add_argument("-ngf", "--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("-ndf", "--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
# parser.add_argument("--seperable_conv", action="store_true", help="Imply seperable conv layer to minify model and calculation")

# hyperparameter
parser.add_argument("--lr", type=float, default=0.00005, help="initial learning rate for adam")

# operating mode

args = parser.parse_args()

# logging configration

# Tensorboard writer
writer = SummaryWriter(args.log_file)

def main():
    G_in = torch.rand((2, 3, 256, 256), requires_grad=False)
    
    G = Generator(in_channels=3 ,args= args)
    D = Discriminator(in_channels=3*2 ,args= args)
    
    G_out = G(G_in)
    D_in = torch.cat((G_in,G_out.detach()), dim=1)
    # D_in = torch.rand((2, 6, 256, 256), requires_grad=False)
    
    writer.add_graph(G, G_in)
    writer.add_graph(D, D_in)

if __name__ == '__main__':
    main()