import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from models import backbone 
from utils.lcls_cfg import cfg
from ActiveWrapper import CIFAR10ActiveWrapper

import warnings
from utils.schedulers import cosine_decay_scheduler

warnings.filterwarnings("ignore")


def load_pretrained_backbone(num_classes : int) -> nn.Module: 
    check_point = torch.load("pretrained_weights/backbone_resnet18_70_sgd_cs_V0.pt")
    model = backbone.BackBoneEncoder(models.__dict__["resnet18"], 2048, 512, in_pretrain=False)

    model.load_state_dict(check_point)
    
    # Freezing the weigths for all the model except the last one
    for name, param in model.named_parameters(): 
        if not name.startswith("encoder.fc") :
            param.requires_grad = False
    
    model.encoder.fc = nn.Sequential(
        nn.Linear(model.last_dim, model.last_dim * 2), 
        nn.BatchNorm1d(model.last_dim * 2),
        nn.ReLU(inplace=True), 
        nn.Linear(model.last_dim * 2, model.last_dim), 
        nn.BatchNorm1d(model.last_dim),
        nn.ReLU(inplace=True), 
        nn.Linear(model.last_dim, num_classes), 
        nn.BatchNorm1d(num_classes, affine=False),
        nn.Softmax(dim = 1)
    )

    return model


def entropy_score(preds) : 

    return  - np.sum(preds * np.log2(preds), axis = 1)



def main() : 

    transforms = T.Compose([
        T.ToTensor()
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cifar10 = datasets.CIFAR10(root="data", train=True, download=False)
    eval_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transforms)

    b_step = int(cfg.b * len(cifar10))

    # eval_indexes = np.random.randint(0, len(eval_data), 5000)
    # eval_data = Subset(eval_data, eval_indexes)


    train_data = CIFAR10ActiveWrapper(cifar10, cfg.initial_budget, cfg.final_budget, cfg.b, transform=transforms)


    loader = DataLoader(train_data, batch_size=cfg.stage2_bs, num_workers=cfg.stage2_num_workers, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=cfg.stage2_bs)


    model = load_pretrained_backbone(10)
    model = model.to(device)

    stage2_optimizer = optim.SGD(model.parameters(), cfg.stage2_lr, cfg.stage2_momentum, weight_decay=cfg.stage2_weigth_decay) 
    stage2_criterion = nn.CrossEntropyLoss()



    # Stage2 Training
    for epoch in range(cfg.stage2_num_epochs):

        model.train()
        train_data.goto_stage2()
        loader = DataLoader(train_data, batch_size=cfg.stage2_bs, num_workers=cfg.stage2_num_workers, shuffle=True)
        
        print("\n")
        print(f"Current budget spent: {(train_data.spent_budget * 100):.2f}% ")
        print(f"Labelled: {len(train_data)}")


        running_loss = 0.0

        for i, (images, labels) in enumerate(loader): 

            images = images.to(device)
            labels = labels.to(device)


            preds = model.forward(images)

            loss = stage2_criterion(preds, labels)

            stage2_optimizer.zero_grad()
            loss.backward()
            stage2_optimizer.step()

            running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss / (i + 1):.4f}", end = "\r")
        

        with torch.no_grad(): 
            model.eval()
            correct = 0
            for images, labels in eval_loader : 

                images = images.to(device)
                labels = labels.to(device)

                preds = model.forward(images) 

                _, predictions = torch.max(preds, 1)
                
                for label, pred in zip(labels, predictions) : 
                    if label == pred: 
                        correct += 1

        
            print(f"\nEpoch {epoch + 1} : Stage2 Finished, Eval Acc: {correct / len(eval_data)}")

            if epoch % 1 == 0 : 

                train_data.goto_stage3()
                loader = DataLoader(train_data, cfg.stage3_bs, cfg.stage3_num_workers)

                history = list()

                # Stage3
                print(f"Epoch {epoch + 1} : Entering Stage3")
                for i, (images, _) in enumerate(loader): 
                    
                    images = images.to(device)
                    preds = model.forward(images)
                    score = entropy_score(preds.cpu().numpy()).tolist()

                    history += score

                    print(f"progress: {i / len(loader) :.2f} ", end = "\r")
                print("")
                _, indices = torch.sort(torch.FloatTensor(history), descending=True)

                train_data.goto_stage3()
            # Dataset still in stage3 mode
                train_data.query_orcale(indices[:b_step])
                cosine_decay_scheduler(stage2_optimizer, .05, epoch, cfg.stage2_num_epochs)
    
if __name__ == "__main__" : 
    main()


