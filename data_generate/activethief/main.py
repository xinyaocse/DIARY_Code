import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.notebook import tqdm, trange

import torch
import torchinfo
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import FGSM, LinfPGD

from src.black_box.model import BlackBoxModel
from src.black_box.oracle import get_oracle_prediction
from src.substitute.datasets import SubstituteDataset, INDICES
from src.substitute.model import SubstituteModel

torch.manual_seed(11)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

p_epochs = 3    # Number of substitute epochs
epochs = 10     # Number of epochs to train the model at each substitute epoch
lr = 1e-2       # Learning rate
lambda_ = 0.1

# STEP 1: Initial Collection
# substitute_dataset = SubstituteDataset(
#     root_dir='src/substitute/data/training_set_10',  # 150 MNIST images here
#     get_predictions=get_oracle_prediction,
#     transform=None
# )

# STEP 2: Architecture Selection, a simple NN
substitute_model = SubstituteModel()
substitute_model.to(device)

for p in trange(p_epochs + 1, desc='Substitute training'):
    # STEP 3: Labeling with oracle, we use get_oracle_prediction to do that, which we
    # treat as a black box in which we only can see the outputs, that is O(x) = label
    substitute_dataset = SubstituteDataset(
        root_dir=f'src/substitute/data/training_set_{p}',
        get_predictions=get_oracle_prediction,
        transform=None
    )
    train_dataloader = DataLoader(
        substitute_dataset,
        batch_size=8,
        shuffle=True
    )

    # STEP 4: Training the substitute model
    substitute_model.train_model(train_dataloader, epochs=epochs, lr=lr)

    # STEP 5: Jacobian dataset augmentation
    substitute_model.jacobian_dataset_augmentation(
        substitute_dataset=substitute_dataset,
        p=(p + 1),
        lambda_=lambda_,
        root_dir=f'src/substitute/data/training_set_{p + 1}',
    )

    # Let's save the model at each substitute epoch p
    torch.save(substitute_model.state_dict(), f'models/substitute_model_p_{p}.pt')

# Final substitute model
torch.save(substitute_model.state_dict(), f'models/substitute_model.pt')