import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as T

import json
import argparse

from torch.utils.data import DataLoader

from torchmetrics.classification import MulticlassAccuracy

from models.backbone import BackboneEncoder
from datasets.RotationLoader import RotationDataset
from utils.meters import AverageMeter


def train(model, criterion, optimizer, loader, epoch, scaler):
    losses = AverageMeter("Loss", ":.4f")

    model.train()

    for batch in loader:
        
        im1, im2, im3, im4 = batch
        img1, rot1 = im1
        img2, rot2 = im2
        img3, rot3 = im3
        img4, rot4 = im4

        images = torch.stack((img1, img2, img3, img4)).view(-1, 3, 32, 32)
        labels = torch.stack((rot1, rot2, rot3, rot4)).view(-1).contiguous()

    

        with torch.autocast(device, dtype=torch.float16):
            images = images.to(device)
            labels = labels.to(device)

            preds = model.forward(images)

            loss = criterion(preds, labels)
        
        losses.update(loss.item(), images.shape[0])

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"Epoch : {epoch} , {losses}")
    return losses.avg

def eval(model, loader, epoch) : 

    model.eval()

    acc = MulticlassAccuracy(num_classes = 4).to(device)

    with torch.no_grad():
        for batch in loader : 
            im1, im2, im3, im4 = batch
            img1, rot1 = im1
            img2, rot2 = im2
            img3, rot3 = im3
            img4, rot4 = im4

            images = torch.stack((img1, img2, img3, img4)).view(-1, 3, 32, 32)
            labels = torch.stack((rot1, rot2, rot3, rot4)).view(-1).contiguous()

            with torch.autocast(device, dtype=torch.float16):
                images = images.to(device)
                labels = labels.to(device)

                preds = model.forward(images)

                acc.update(preds, labels)
    
    return acc.compute.item()
            


def main():

    with open(args.config, "r") as f:
        config = json.load(f)

    transforms = T.Compose(
        [
            # Maybe add some more augmentations ??? Remember not to use any Flip !!
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    _train_data = datasets.CIFAR10(root="data", train=True, download=True)

    train_data = RotationDataset(_train_data, transforms=transforms)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    _eval_data = datasets.CIFAR10(root='data', train=False, download=True)
    eval_data = RotationDataset(_eval_data, transforms=transforms)
    eval_loader = DataLoader(
        eval_data, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    model = BackboneEncoder(4).to(device)

    optim_config = config["optimizer"]

    # Is SGD the best choice ... have to test adam too
    optimizer = optim.SGD(
        model.parameters(),
        lr=optim_config["lr"],
        momentum=optim_config["momentum"],
        weight_decay=optim_config["weight_decay"],
    )

    current_epoch = 0
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=[ 0.3 * args.num_epochs , .6 * args.num_epochs],
                                                     gamma=.2,
                                                     verbose=True)
    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    loss_history = list()
    acc_history = list()

    if config["checkpoint"]:

        checkpoint = torch.load(config["path_to_checkpoint"])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        current_epoch = checkpoint["epoch"]
        loss_history = checkpoint["loss"]
        acc_history = checkpoint["acc"]

        print(f"Resuming Training from Epoch {current_epoch}, Last Loss {loss_history[-1]}")

    
 

    for epoch in range(current_epoch, args.num_epochs):
        print(f"Epoch {epoch}")
        avg_epoch_loss = train(model, criterion, optimizer, train_loader, epoch, scaler)
        scheduler.step()

        eval_acc = eval(model, eval_loader, epoch )
        print(f"Acc on Eval {eval_acc}")
        acc_history.append(eval_acc)

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss":loss_history.append(avg_epoch_loss),
                "acc" : acc_history
            },
            config["path_to_checkpoint"],
        )
        config["checkpoint"] = True

        with open(args.config, "w") as out:
            json.dump(config, out, indent=4)
        print("Checkpoint Saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain the backbone")

    parser.add_argument(
        "--config",
        help="path to the json config file",
        type=str,
        default="pretrain.config.json",
    )

    parser.add_argument(
        "--batch_size",
        help="Batch Size : Note it will be x4 on memory",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--num_workers", help="Number of jobs to spawn", type=int, default=1
    )

    parser.add_argument(
        "--num_epochs", help="Number of training epochs", type=int, default=100
    )

    parser.add_argument(
        "--device", help="Device on which to run the training", type=str, default="gpu"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu"
    print(f"[ + ] Device set to: {device}")

    main()
