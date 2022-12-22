import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
import pandas as pd
import warnings
import argparse
import json

from torch.nn.init import kaiming_uniform_, normal_
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from utils.meters import AverageMeter
from ActiveWrapper import CIFAR10ActiveWrapper

from utils.schedulers import cosine_decay_scheduler

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    help="path to the config file, any command line argument will override the config file",
    type=str,
)
parser.add_argument(
    "--device", help="Whant device to use", choices=["cpu", "gpu"], type=str
)

parser.add_argument(
    "--num_epochs",
    help="Total number of epoch for pretraining path to checkpoint in config file",
    type=int,
)

parser.add_argument(
    "--base_line_eval", 
    help="Wether to apply active learning techiques or nor", 
    type=bool, 
    default=False
)
parser.add_argument("--batch_size", help="Batch size", type=int)
parser.add_argument("--num_workers", help="how many workers to spwan for the dataloader", type=int, default=1)

args = parser.parse_args()

with open(args.config, "r") as f: 
    config = json.load(f)

if args.base_line_eval : 
    print("[ + ] Baseline evaluation, Using whole dataset, No stage 3")


def load_pretrained_backbone(num_classes: int) -> nn.Module:
    """
    Load the resnet backbone with just a linear layer at the end.
    Freeze all the weights excpet for the last layer.
    load from checkpoint.
    """

    checkpoint = torch.load(config["path_to_checkpoint"])
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

    transforms_train = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(), 
        T.RandomApply([T.ColorJitter(.4, .4, .4, .4)], p=.2)
    ])

    transforms_eval = T.Compose([
        T.ToTensor()
    ])

    device = "cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu"
    eval_data = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transforms_eval
    )



    # How many images to add to add to the labeled set evry cycle
    if not args.base_line_eval: 
        cifar10 = datasets.CIFAR10(root="data", train=True, download=False)
        train_data = CIFAR10ActiveWrapper(
            cifar10, config["al"]["initial_budget"], config["al"]["final_budget"], config["al"]["to_label_each_round"], transform=transforms_train
        )
        b_step = int( config["al"]["to_label_each_round"] * len(cifar10))
    else: 
        train_data = datasets.CIFAR10(root="data", train=True, download=False, transform=transforms_train)


    # loader = DataLoader(
    #     train_data,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     shuffle=True,
    # )

    eval_loader = DataLoader(eval_data, batch_size=args.batch_size)

    model = load_pretrained_backbone(10)
    model = model.to(device)

    # The optimizer should work only on the non-frozen modules
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert(len(parameters) == 2)

    stage2_optimizer = optim.SGD(
        parameters,
        config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
        momentum=config["optimizer"]["weight_decay"],
    )
    
    stage2_criterion = nn.CrossEntropyLoss()

    # Stage2 Training
    for epoch in range(args.num_epochs):

        model.train()

        if not args.base_line_eval:
            train_data.goto_stage2()

        loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

        if not args.base_line_eval:
            print("\n")
            print(f"Current budget spent: {(train_data.spent_budget * 100):.2f}% ")
            print(f"Labelled: {len(train_data)}")

        losses = AverageMeter("Loss", ":.5f")

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

            # running_loss += loss.item()
            losses.update(loss.item(), images.size(0))

            print(f"Epoch {epoch + 1}, {losses}" , end="\r")

        cosine_decay_scheduler(
            stage2_optimizer, config["optimizer"]["lr"], epoch, args.num_epochs
        )
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



            # Asking the oracle step untill budget is exhausted
            if not args.base_line_eval and epoch % 1 == 0 and train_data.spent_budget < config["al"]["final_budget"]:

                train_data.goto_stage3()
                loader = DataLoader(train_data, args.batch_size, args.num_workers)

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


if __name__ == "__main__":
    main()
