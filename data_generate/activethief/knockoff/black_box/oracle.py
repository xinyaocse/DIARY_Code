"""
Trained black-box model that will be used as an oracle.
"""
import os

import torch
from torch import Tensor

from .model import BlackBoxModel
from.resnet import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = f"{os.path.dirname(os.path.realpath(__file__))}/ckpt.pth"

ORACLE = ResNet18()
checkpoint = torch.load(MODEL_PATH)
ORACLE.load_state_dict(checkpoint['net'],strict=False)
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
# ORACLE.load_state_dict(torch.load(MODEL_PATH))
ORACLE.to(device)
ORACLE.eval()


def get_oracle_prediction(x: Tensor):
    """

    Paramaters
    ----------
    x: Tensor.
        PyTorch tensor representing an MNIST image or batch of images
        in the range (0,1).

    Returns
    -------
    Tensor, labels of the predicted images using the oracle.
    """
    x = x.to(device)
    return ORACLE(x).argmax(dim=-1)
