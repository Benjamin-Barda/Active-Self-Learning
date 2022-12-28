import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset, Subset, IterableDataset, DataLoader
from torchvision import transforms as T
from collections import Counter

import numpy as np

import random


class RotationDataset(Dataset):

    """
    Currently support only CIFAR10
    """

    def __init__(self, dataset, transforms=None) -> None:

        if not isinstance(dataset, datasets.CIFAR10):
            raise TypeError(
                f"Error loading dataset: Expected type datasets.CIFAR10 but got {type(dataset)}"
            )

        self.dataset = dataset
        self.transforms = transforms
    
    def collate_fn(self, batch) : 

        

        im1, im2, im3, im4 = batch
        img1, rot1 = im1
        img2, rot2 = im2
        img3, rot3 = im3
        img4, rot4 = im4

        images = torch.stack((img1, img2, img3, img4)).view(-1, 3, 32, 32)
        labels = torch.stack((rot1, rot2, rot3, rot4)).view(-1)

        return images, labels
       

        

    def __len__(self) -> int:
        return len(self.dataset)
    

    def __getitem__(self, index):
        """
        labels : 0 -> no rotations
                 1 -> 90deg
                 2 -> 180deg
                 3 -> 270deg
        """
        img, _ = self.dataset[index]

        if self.transforms :
            img = self.transforms(img)

        img90 = torch.rot90(img, 1, [1, 2])
        img180 = torch.rot90(img, 2, [1, 2])
        img270 = torch.rot90(img, 3, [1, 2])

        imgs = [img, img90, img180, img270]
        rotations = [0, 1, 2, 3]

        random.shuffle(rotations)

        return (imgs[rotations[0]], rotations[0]),(imgs[rotations[1]], rotations[1]),(imgs[rotations[2]], rotations[2]),(imgs[rotations[3]], rotations[3])


if __name__ == "__main__" : 

    show = False

    import matplotlib.pyplot as plt

    data = datasets.CIFAR10(root="data", train=True, download=False)
    trans = T.Compose([
        T.ToTensor()
    ])

    rot_data = RotationDataset(data, trans) 

    loader = DataLoader(rot_data, batch_size=128)

    for batch in loader : 

        im1, im2, im3, im4 = batch
        img1, rot1 = im1
        img2, rot2 = im2
        img3, rot3 = im3
        img4, rot4 = im4


        images = torch.stack((img1, img2, img3, img4)).view(-1, 3, 32, 32)
        labels = torch.stack((rot1, rot2, rot3, rot4)).view(-1)

        print(images.shape)
        break

        
    

    if show: 
        imgs, _ = rot_data[100]
        
        a, b, c, d = imgs

        fig, ax = plt.subplots(nrows=2, ncols=2)

        ax[0][0].imshow(a.permute(1,2,0).numpy())
        ax[0][1].imshow(b.permute(1,2,0).numpy())
        ax[1][0].imshow(c.permute(1,2,0).numpy())
        ax[1][1].imshow(d.permute(1,2,0).numpy())

        plt.show()



