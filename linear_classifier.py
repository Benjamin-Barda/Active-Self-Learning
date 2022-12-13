import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
import pandas as pd
import warnings
import argparse

from torch.nn.init import kaiming_uniform_, normal_
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from utils.lcls_cfg import cfg
from ActiveWrapper import CIFAR10ActiveWrapper

from utils.schedulers import cosine_decay_scheduler

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

base_line_eval = True
if base_line_eval : 
    print("[ + ] Baseline evaluation, Using whole dataset, No stage 3")

args = parser.parse_args()

def load_pretrained_backbone(num_classes: int) -> nn.Module:
    """
    Load the resnet backbone with just a linear layer at the end.
    Freeze all the weights excpet for the last layer.
    load from checkpoint.
    """

    checkpoint = torch.load("checkpoint/resnet18_533_sgd_V0.pt")
    model_state = checkpoint["model"]

    model = models.resnet18(num_classes=num_classes)

    fc_weigth = kaiming_uniform_(model.fc.weight.data)
    fc_bias = normal_(model.fc.bias.data)

    # Rebuild state_dic from checkpoint with correct keys
    state_dict = dict()

    # prefix to be removed
    offset_char = "encoder."

    state_dict = {
        k[len(offset_char) :]: model_state[k]
        for k in model_state.keys()
        if k[len(offset_char) :] in model.state_dict().keys()
    }
    state_dict["fc.weight"] = fc_weigth
    state_dict["fc.bias"] = fc_bias

    model.load_state_dict(state_dict)

    # Freezing the weigths for all the model except the last one
    for name, param in model.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False

    return model


def entropy_score(preds):
    # preds (bs, 10)
    preds = nn.functional.softmax(preds, dim=1)
    return -torch.sum(preds * torch.log2(preds), dim=1)


def visualize_ds_stats(labels):
    labels_df = pd.DataFrame(labels.data)


def main():

    transforms = T.Compose([T.ToTensor()])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_data = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transforms
    )

    # How many images to add to add to the labeled set evry cycle
    if not base_line_eval: 
        cifar10 = datasets.CIFAR10(root="data", train=True, download=False)
        train_data = CIFAR10ActiveWrapper(
            cifar10, cfg.initial_budget, cfg.final_budget, cfg.b, transform=transforms
        )
        b_step = int(cfg.b * len(cifar10))
    else: 
        train_data = datasets.CIFAR10(root="data", train=True, download=False, transform=transforms)


    loader = DataLoader(
        train_data,
        batch_size=cfg.stage2_bs,
        num_workers=cfg.stage2_num_workers,
        shuffle=True,
    )

    eval_loader = DataLoader(eval_data, batch_size=cfg.stage2_bs)

    model = load_pretrained_backbone(10)
    model = model.to(device)

    stage2_optimizer = optim.SGD(
        model.parameters(),
        cfg.stage2_lr,
        weight_decay=cfg.stage2_weigth_decay,
        momentum=0.9,
    )
    stage2_criterion = nn.CrossEntropyLoss()

    # Stage2 Training
    for epoch in range(cfg.stage2_num_epochs):

        model.train()

        if not base_line_eval:
            train_data.goto_stage2()

        loader = DataLoader(
            train_data,
            batch_size=cfg.stage2_bs,
            num_workers=cfg.stage2_num_workers,
            shuffle=True,
        )

        if not base_line_eval:
            print("\n")
            print(f"Current budget spent: {(train_data.spent_budget * 100):.2f}% ")
            print(f"Labelled: {len(train_data)}")

        running_loss = 0.0

        # Traning Loop
        for i, (images, labels) in enumerate(loader):

            images = images.to(device)

            labels = nn.functional.one_hot(labels, 10).type(torch.FloatTensor)
            labels = labels.to(device)

            preds = model.forward(images)

            stage2_optimizer.zero_grad()
            loss = stage2_criterion(preds, labels)
            loss.backward()
            stage2_optimizer.step()

            running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss /(i + 1) }", end="\r")
        # Eval
        with torch.no_grad():

            model.eval()
            total = 0
            correct = 0

            for images, labels in eval_loader:

                images = images.to(device)
                labels = labels.to(device)

                preds = model.forward(images)
                total += labels.size(0)
                _, predictions = torch.max(preds.data, 1)
                correct += (predictions == labels).sum().item()

            print(
                f"\nEpoch {epoch + 1} : Stage2 Finished, Eval Acc: {100 * correct // total}%"
            )


            # Asking the oracle step untill budget is exhausted
            if not base_line_eval and epoch % 1 == 0 and train_data.spent_budget < cfg.final_budget:

                train_data.goto_stage3()
                loader = DataLoader(train_data, cfg.stage3_bs, cfg.stage3_num_workers)

                history = list()
                # Stage3
                print(f"Epoch {epoch + 1} : Entering Stage3")
                for i, (images, _) in enumerate(loader):

                    images = images.to(device)
                    preds = model.forward(images)
                    score = entropy_score(preds.cpu()).tolist()

                    history += score

                    print(f"progress: {i / len(loader) :.2f} ", end="\r")
                print("")
                _, indices = torch.sort(torch.FloatTensor(history), descending=True)

                # Dataset still in stage3 mode
                train_data.query_oracle_r(indices[:b_step])
                cosine_decay_scheduler(
                    stage2_optimizer, 0.05, epoch, cfg.stage2_num_epochs
                )


if __name__ == "__main__":
    main()
