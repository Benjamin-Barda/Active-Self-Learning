import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np

from torch.utils.data import DataLoader

from models import backbone 
from utils.lcls_cfg import cfg



def get_initial_subset(initial_budget : float, ds_size : int, seed = 0) -> tuple[list[int], map[int, float]] :
    np.random.seed(seed)
    expected_size = int(ds_size * initial_budget)
    if expected_size >= ds_size : 
        raise ValueError("Initial Budget Too big")
    
    index_map = {k : False for k in range(expected_size)}
    stage2_index = list()
    stage3_index = list()

    while len(stage2_index) != expected_size:  
        pick = np.random.randint(0, ds_size, 1 )
        if not index_map[pick] : 
            stage2_index.append(pick)
            index_map[pick] = True
    
    for k, v in index_map.items() : 
        if not v :
            stage3_index.append(k)
        
    return (stage2_index, stage3_index, index_map)


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    whole_data = datasets.CIFAR10(root="data", train=True, download=True)
    stage2_indexes, stage3_indexes, index_map = get_initial_subset(cfg.initial_budget, len(whole_data))

    # Labelled
    stage2_data = torch.utils.data.Subset(whole_data, stage2_indexes)

    # Non-Labelled
    stage3_data = torch.utils.data.Subset(whole_data, stage3_indexes)

    # Labelled | We use the whole evaluation set to measure the classification performances. 
    eval_data = datasets.CIFAR10(root="data", train=False, download=True)

    # Backbone + Classification Head | Weight frozen on all except last layer
    model = load_pretrained_backbone(10)
    model = model.to(device)

    stage2_train_loader = DataLoader(stage2_data, batch_size=cfg.stage2_bs, shuffle=True)
    stage2_eval_loader = DataLoader()
    stage2_optimizer = optim.SGD(model.parameters(), cfg.stage2_lr, cfg.stage2_momentum, weight_decay=cfg.stage2_weigth_decay) 
    stage2_criterion = nn.CrossEntropyLoss()

    stage3_loader = DataLoader(stage3_data, batch_size=cfg.stage3_bs, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=cfg.stage2_bs, shuffle=True)




    
    
if __name__ == "__main__" : 
    main()

