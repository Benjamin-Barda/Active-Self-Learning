import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import json
import argparse

from torch.utils.data import DataLoader, Subset

from models.backbone import BackBoneEncoder
from utils.schedulers import cosine_decay_scheduler
from utils.meters import AverageMeter

parser = argparse.ArgumentParser(description="Pretrain the backbone")

parser.add_argument(
    "--config",
    help="path to the config file, any command line argument will override the config file",
    type=str,
)
parser.add_argument(
    "--device", help="Whant device to use", choices=["cpu", "gpu"], type=str
)
parser.add_argument(
    "--model",
    help="Choose the backbone for the simsiam network",
    choices=["resnet18", "resnet50"],
    type=str,
)
parser.add_argument("--encoder_dim", help="Output dimension for the encoder", type=int)
parser.add_argument("--pred_dim", help="Output dimension for the predictor", type=int)
parser.add_argument("--batch_size", help="Batch size", type=int)
parser.add_argument("--num_epochs", help="Total number of epoch for pretraining path to checkpoint in config file", type=int)


args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu"
print(f"[ + ] Device set to: {device}")


class TwoCropTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, image):
        q = self.base_transform(image)
        k = self.base_transform(image)

        return [q, k]


def train(model, criterion, optimizer, loader, epoch):
    losses = AverageMeter("Loss", ":.4f")

    for _, (images, _) in enumerate(loader):

        images[0] = images[0].to(device, non_blocking=True)
        images[1] = images[1].to(device, non_blocking=True)

        p1, p2, z1, z2 = model(images[0], images[1])

        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), images[0].shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch : {epoch} , {losses}")
    return losses.avg


def main():

    with open(args.config, "r") as f:
        config = json.load(f)


    model = BackBoneEncoder(
        models.__dict__[args.model],
        args.encoder_dim,
        args.pred_dim,
        in_pretrain=True,
    ).to(device)

    augmentation = [
        transforms.RandomResizedCrop(28, scale=(0.2, 1)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=2),
        transforms.ToTensor(),
    ]

    batch_size = args.batch_size
    current_epoch = 0
    num_epochs = args.num_epochs

    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=TwoCropTransform(transforms.Compose(augmentation)),
    )

    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    criterion = nn.CosineSimilarity(dim=1).to("cuda")

    optim_config = config["optimizer"]
    optimizer = optim.SGD(
        model.parameters(),
        lr=optim_config["lr"],
        momentum=optim_config["momentum"],
        weight_decay=optim_config["weight_decay"],
    )

    if config["checkpoint"]:

        checkpoint = torch.load(config["path_to_checkpoint"])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        current_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

        print(f"Resuming Training from Epoch {current_epoch}, Last Loss {loss}")

    model.train()

    for name, param in model.named_parameters(): 
        print(name, param.requires_grad) 
    

    for epoch in range(current_epoch, num_epochs):
        print(f"Epoch {epoch}")
        avg_epoch_loss = train(model, criterion, optimizer, loader, epoch)
        cosine_decay_scheduler(optimizer, 0.05, epoch, num_epochs)

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_epoch_loss,
            },
            config["path_to_checkpoint"],
        )
        config["checkpoint"] = True

        with open(args.config, "w") as out:
            json.dump(config, out, indent=4)
        print("Checkpoint Saved")


if __name__ == "__main__":
    main()
