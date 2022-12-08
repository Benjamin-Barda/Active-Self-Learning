import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np

from torch.utils.data import DataLoader

from models import backbone 
from utils.lcls_cfg import cfg


train_data = datasets.CIFAR10(root="data", train=True, download=False) 

def build_initial_labels(initial_budget : float, ds_size : int, seed = 0) -> np.ndarray[int] :
    np.random.see(seed)
    expected_size = int(ds_size * initial_budget)
    if expected_size >= ds_size : 
        raise ValueError("Initial Budget Too big")

    initial_pick = np.random.randint(0, ds_size, expected_size)
    print(f"[ + ] Picked {initial_pick} samples")
    return initial_pick


def load_pretrained_backbone(num_classes : int): 
    check_point = torch.load("pretrained_weights/backbone_resnet18_70_sgd_cs_V0.pt")
    model = backbone.BackBoneEncoder(models.__dict__["resnet18"], 2048, 512)

    model.load_state_dict(check_point)
    
    for name, param in model.named_parameters(): 
        if not name.startswith("encoder.fc") :
            param.requires_grad = False
    
    model.encoder.fc = nn.Sequential(
        nn.Linear(model.last_dim, model.last_dim), 
        nn.BatchNorm1d(model.last_dim),
        nn.Mish(inplace=True), 
        nn.Linear(model.last_dim, num_classes), 
        nn.BatchNorm1d(num_classes),
        nn.Softmax(num_classes)
    )

    return model
    
def stage2():
    '''
    In this stage we train the classifier on the data that is labeled thus far
    ''' 
    ...

def stage3():
    '''
    In this stage we pass over the unlabeled data we then query the "oracle" to label the topK performing samples based on some acquisition function.
    ''' 
    ...