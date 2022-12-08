import torch
import torch.optim as optim
import torch.nn as nn

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from models.backbone import BackBoneEncoder
from utils.backbone_cfg import cfg
from utils.schedulers import cosine_decay_scheduler
from utils.meters import AverageMeter

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[ + ] Device set to: {device}")

def load_data() -> DataLoader: 

  augmentation = [
      transforms.RandomResizedCrop(28, scale=(.2, 1)),
      transforms.RandomApply([transforms.ColorJitter(.4, .4, .4, .1)], p = .6),
      transforms.RandomHorizontalFlip(),
      transforms.RandomGrayscale(p = 2),
      transforms.ToTensor()
  ]

  class TwoCropTransform : 

    def __init__(self, base_transform) : 
      self.base_transform = base_transform

    def __call__(self, image) : 
      q = self.base_transform(image)
      k = self.base_transform(image)

      return [q, k]

  train_data = datasets.CIFAR10(root = "data", train = True, download = True, transform = TwoCropTransform(transforms.Compose(augmentation)))
  loader = DataLoader(train_data, batch_size = cfg.batch_size, shuffle=True)

  return loader


def train(model, criterion, optimizer, loader, epoch):
  losses = AverageMeter("Loss",  ":.4f") 

  for _, (images, _) in enumerate(loader):

    images[0] = images[0].to(device, non_blocking = True)
    images[1] = images[1].to(device, non_blocking = True)
    
    p1, p2, z1, z2 = model(images[0], images[1])

    loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

    losses.update(loss.item(), cfg.batch_size)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f"Epoch : {epoch} , {losses}")



def main(): 

  loader = load_data()
  model = BackBoneEncoder(models.__dict__[cfg.backbone_arch], cfg.backbone_dim, cfg.backbone_pred_dim, in_pretrain=True).to(device)

  criterion = nn.CosineSimilarity(dim = 1).to("cuda")
  optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum = cfg.momentum, weight_decay=cfg.weight_decay)

  for epoch in cfg.num_epochs:
    train(model, criterion, optimizer, loader, epoch)
    cosine_decay_scheduler(optimizer, .05, epoch, cfg.num_epochs )
  
  torch.save(model.state_dict(), "resnet18_70_V0.pt")

if __name__ == "__main__" :
  main()