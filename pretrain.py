import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models import resnet18

from dataset.RotationLoader import RotationDataset
from utils.meters import AverageMeter


def train(model, criterion, optimizer, loader, epoch, scaler):

    losses = AverageMeter("Loss", ":.4f")
    model.train()

    for im1, im2, im3, im4, tg1, tg2, tg3, tg4 in loader:

        with torch.autocast(device, dtype=torch.float16):

            im1, im2, im3, im4 = (
                im1.to(device),
                im2.to(device),
                im3.to(device),
                im4.to(device),
            )
            tg1, tg2, tg3, tg4 = (
                tg1.to(device),
                tg2.to(device),
                tg3.to(device),
                tg4.to(device),
            )

            pr1, pr2, pr3, pr4 = model(im1), model(im2), model(im3), model(im4)

            l1 = criterion(pr1, tg1)
            l2 = criterion(pr2, tg2)
            l3 = criterion(pr3, tg3)
            l4 = criterion(pr4, tg4)

            # This is the actual paper implementation. I do not beleive it makes any changes.
            loss = (l1 + l2 + l3 + l4) / 4

        losses.update(loss.item(), im1.shape[0] * 4)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"Epoch : {epoch} , {losses}")
    return losses.avg


def eval(model, loader, epoch):

    model.eval()

    acc = MulticlassAccuracy(num_classes=4).to(device)

    with torch.no_grad():

        for batch in loader:
            img1, img2, img3, img4, rot1, rot2, rot3, rot4 = batch

            images = torch.stack((img1, img2, img3, img4)).view(-1, 3, 32, 32)
            labels = torch.stack((rot1, rot2, rot3, rot4)).view(-1).contiguous()

            with torch.autocast(device, dtype=torch.float16):
                images = images.to(device)
                labels = labels.to(device)

                preds = model.forward(images)

                acc.update(preds, labels)

    return acc.compute().item()


def main():

    with open(args.config, "r") as f:
        config = json.load(f)
    train_transforms = T.Compose(
        [
            # Maybe add some more augmentations ???
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    eval_transforms = T.Compose(
        [
            # Maybe add some more augmentations ???
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    _train_data = datasets.CIFAR10(root="data", train=True, download=True)

    train_data = RotationDataset(_train_data, transforms=train_transforms)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    _eval_data = datasets.CIFAR10(root="data", train=False, download=True)
    eval_data = RotationDataset(_eval_data, transforms=eval_transforms)
    eval_loader = DataLoader(
        eval_data, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = resnet18()
    out_dim = model.fc.weight.shape[1]
    model.fc = nn.Linear(out_dim, 4)
    model.to(device)

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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 50, 70, 90],
        verbose=False,
    )
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

        print(
            f"Resuming Training from Epoch {current_epoch}, Last Loss {loss_history[-1]}"
        )

    for epoch in range(current_epoch, args.num_epochs):
        print(f"Epoch {epoch}")

        avg_epoch_loss = train(model, criterion, optimizer, train_loader, epoch, scaler)
        scheduler.step()
        loss_history.append(avg_epoch_loss)

        eval_acc = eval(model, eval_loader, epoch)
        print(f"Acc on Eval {eval_acc}")
        acc_history.append(eval_acc)

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": loss_history,
                "acc": acc_history,
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
        default=256,
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
