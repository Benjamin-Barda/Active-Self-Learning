import argparse
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms as T

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy

from dataset.ActiveWrapper import CIFAR10ActiveWrapper
from utils.meters import AverageMeter

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    help="path to the config file, any command line argument will override the config file",
    type=str,
    default="lcls.config.json",
)
parser.add_argument(
    "--device",
    help="Whant device to use",
    choices=["cpu", "gpu"],
    type=str,
    default="gpu",
)

parser.add_argument(
    "--num_epochs",
    help="Total number of epoch for pretraining path to checkpoint in config file",
    type=int,
    default=500,
)

parser.add_argument(
    "--base_line_eval",
    help="Wether to apply active learning techiques or nor",
    type=bool,
    default=False,
)
parser.add_argument("--batch_size", help="Batch size", type=int, default=256)
parser.add_argument(
    "--num_workers",
    help="how many workers to spwan for the dataloader",
    type=int,
    default=1,
)

parser.add_argument(
    "--cycle_len",
    help="How many epochs before querying the oracle",
    type=int,
    default=50,
)

parser.add_argument(
    "--freeze",
    help="Whether to freeze the pretrained backbone",
    type=bool,
    default=False,
)

args = parser.parse_args()

with open(args.config, "r") as f:
    config = json.load(f)

if args.base_line_eval:
    print("[ + ] Baseline evaluation, Using whole dataset, No stage 3")


def load_pretrained_backbone(num_classes: int, get_pretext_losses : bool = False) -> nn.Module:
    """
    Load the resnet backbone with just a linear layer at the end.
    Freeze all the weights excpet for the last layer.
    load from checkpoint.
    """



    checkpoint = torch.load(config["path_to_checkpoint"], map_location='cpu')
    model_state = checkpoint["state_dict"]

    model = models.resnet18(num_classes=num_classes)

    new_d = dict()

    for k, v in model_state.items():
        if k.startswith("backbone."):
            new_k = k.replace("backbone.", "")
            new_d[new_k] = v

    model.load_state_dict(new_d, strict=False)

    model.fc = nn.Linear(512, num_classes)

    # Freezing the weigths for all the model except the last one
    if args.freeze:
        for name, param in model.named_parameters():
            if name not in ["fc.weight", "fc.bias"] :
                param.requires_grad = False
            
    return model


def entropy_score(preds):
    # preds (bs, 10)
    preds = nn.functional.softmax(preds, dim=1)
    return - torch.sum(preds * torch.log2(preds), dim=1)

def min_margin(preds): 
    preds = torch.sort(preds, dim=1)



def main():

    writer = SummaryWriter()

    transforms_train = T.Compose(
        [
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            T.RandomCrop(32, padding=4),
            T.GaussianBlur((3)), 
            T.ColorJitter(.1, .1, .1, .1)
        ]
    )

    transforms_eval = T.Compose(
        [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    device = "cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu"
    eval_data = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transforms_eval
    )

    # How many images to add to add to the labeled set evry cycle
    if not args.base_line_eval:
        cifar10 = datasets.CIFAR10(root="data", train=True, download=False)
        train_data = CIFAR10ActiveWrapper(
            cifar10,
            config["al"]["initial_budget"],
            config["al"]["final_budget"],
            config["al"]["to_label_each_round"],
            transform=transforms_train,
        )
        b_step = int(config["al"]["to_label_each_round"] * len(cifar10))
    else:
        train_data = datasets.CIFAR10(
            root="data", train=True, download=False, transform=transforms_train
        )

    eval_loader = DataLoader(eval_data, batch_size=args.batch_size)

    if args.base_line_eval:
        model = models.resnet18()
        model = model.to(device)
    else:
        model = load_pretrained_backbone(10)
        model = model.to(device)

    stage2_optimizer = optim.SGD(
        model.parameters(),
        config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
        nesterov=True, 
        momentum=config ['optimizer']['momentum']
    )

    # stage2_optimizer = optim.AdamW(
    #     model.parameters(),
    #     config["optimizer"]["lr"],
    #     weight_decay=config["optimizer"]["weight_decay"],
    #     amsgrad=True,
    # )

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        stage2_optimizer,
        T_max=args.num_epochs
    )


    acc = MulticlassAccuracy(num_classes=10).to(device)

    # Tracker for results

    if not args.base_line_eval :
        

        first_batch_losses = list()
        cf = datasets.CIFAR10(root="data", train=True, transform=transforms_eval)  
        loader = DataLoader(cf, batch_size=500, shuffle=False)
        initial_sample_size = config["al"]["initial_budget"] * len(cifar10)

        model.eval() 

        for img, _ in loader : 
        
            img = img.to(device)
        
            preds = model(img)

            first_batch_losses += entropy_score(preds.cpu()).tolist()
        
        _, indices = torch.sort(torch.FloatTensor(first_batch_losses), descending=True)

        train_data.goto_stage3()
        train_data.query_oracle(indices[:int(initial_sample_size)])
        


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
            writer.add_scalar("Labelled", (train_data.spent_budget * 100), epoch )
            print(f"Labelled: {len(train_data) }")

        losses = AverageMeter("Loss", ":.5f")

        # Traning Loop
        for i, (images, labels) in enumerate(loader):

            stage2_optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)
            preds = model.forward(images)

            loss = criterion(preds, labels)
            loss.backward()

            stage2_optimizer.step()

            # running_loss += loss.item()
            losses.update(loss.item(), images.size(0))

            print(f"Epoch {epoch + 1}, {losses}", end="\r")

        writer.add_scalar("Loss/Train", losses.avg, epoch)
        scheduler.step()
        losses.reset()

        # Eval
        with torch.no_grad():

            model.eval()
            total = 0
            correct = 0

            for images, labels in eval_loader:

                images = images.to(device)
                labels = labels.to(device)

                preds = model.forward(images)

                loss = criterion(preds, labels)
                losses.update(loss.item(), images.size(0))

                acc.update(preds, labels)
                total += labels.size(0)

                _, predictions = torch.max(preds.data, 1)
                correct += (predictions == labels).sum().item()

            print(
                f"\nEpoch {epoch + 1} : Stage2 Finished, Eval Acc: {acc.compute().item()}%"
            )
            writer.add_scalar("Loss/Eval", losses.avg, epoch)
            writer.add_scalar("Accuracy/Eval", acc.compute().item(), epoch)
            acc.reset()

            # Asking the oracle step untill budget is exhausted
            if (
                not args.base_line_eval
                and epoch % args.cycle_len == args.cycle_len - 1
                and train_data.spent_budget <= config["al"]["final_budget"]
            ):

                train_data.goto_stage3()
                loader = DataLoader(train_data, args.batch_size, args.num_workers)

                history = list()

                # Stage3 - Acquisition function and query stage
                print(f"Epoch {epoch + 1} : Entering Stage3")
                for i, (images, _) in enumerate(loader):

                    images = images.to(device)
                    preds = model.forward(images)
                    score = entropy_score(preds.cpu()).tolist()

                    history += score

                    print(f"progress: {i / len(loader) :.2f} ", end="\r")

                # Sort Based on Entropy Score
                _, indices = torch.sort(torch.FloatTensor(history), descending=True)

                # Dataset still in stage3 mode
                train_data.query_oracle(indices[:b_step])

                
    writer.close()

if __name__ == "__main__":
    main()
