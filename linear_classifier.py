import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms as T

from models import backbone 
from utils.lcls_cfg import cfg
from ActiveWrapper import CIFAR10ActiveWrapper






def load_pretrained_backbone(num_classes : int) -> nn.Module: 
    check_point = torch.load("pretrained_weights/backbone_resnet18_70_sgd_cs_V0.pt")
    model = backbone.BackBoneEncoder(models.__dict__["resnet18"], 2048, 512, in_pretrain=False)

    model.load_state_dict(check_point)
    
    # Freezing the weigths for all the model except the last one
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


def query_oracle(stage2_data, stage3_data, index_map) : 
    '''
    Pass cfg.b images from stage3 to stage stage2 data adding labels
    '''
    ...




def main() : 

    transforms = T.Compose([
        T.ToTensor()
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cifar10 = datasets.CIFAR10(root="data", train=True, download=False)
    eval_data = datasets.CIFAR10(root="data", train=False, download=True)


    train_data = CIFAR10ActiveWrapper(cifar10, cfg.initial_budget, cfg.final_budget, transform=transforms)


    stage2_Loader = train_data.get_stage2_loader(cfg.stage2_bs, cfg.stage2_num_workers)
    eval_loader = DataLoader(eval_data, batch_size=cfg.stage2_bs, shuffle=True)

    stage3_Loader = train_data.get_stage3_loader(cfg.stage3_bs, cfg.dstage3_num_workers)

    model = load_pretrained_backbone(10)
    model = model.to(device)

    stage2_optimizer = optim.SGD(model.parameters(), cfg.stage2_lr, cfg.stage2_momentum, weight_decay=cfg.stage2_weigth_decay) 
    stage2_criterion = nn.CrossEntropyLoss()

    
    
if __name__ == "__main__" : 
    main()

