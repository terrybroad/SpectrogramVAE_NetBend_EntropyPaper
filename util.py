import torch
from torch import nn, optim
from torch.autograd import Variable, grad
import numpy as np

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
