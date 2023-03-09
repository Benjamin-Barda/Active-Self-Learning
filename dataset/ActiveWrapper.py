import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset, Subset, IterableDataset, DataLoader
from torchvision import transforms
from collections import Counter

import numpy as np

import random

class CIFAR10ActiveWrapper(Dataset):

    STAGE2 = 2
    STAGE3 = 3

    """
    CIFAR10 wrapper for active learning
    """

    def __init__(
        self,
        dataset: datasets.CIFAR10,
        initial_budget: float,
        total_budget: float,
        b: int,
        initial_stage: int = 2,
        transform: transforms = None,
        seed : int = None
    ):
        """
        Args:
            dataset (datasets.CIFAR10)
            initial_budget (float) :
            total_budget (float) :
            b (int) :
            initial_stage (int) :
            transform (transforms) :
        """
        possible_stages = [self.STAGE2, self.STAGE3]

        if initial_stage not in possible_stages:
            raise ValueError(f"Error in stage: expected 2 or 3 but got {initial_stage}")

        if not isinstance(dataset, datasets.CIFAR10):
            raise TypeError(
                f"Error loading dataset: Expected type datasets.CIFAR10 but got {type(dataset)}"
            )

        self.dataset = dataset
        self.current_stage = initial_stage

        # TODO Change them to numpy array
        self.stage2_indexes = list()
        self.stage3_indexes = list()

        if total_budget > 1:
            raise ValueError(
                f"Error in budget processing: Expected in range [0, 1] but got {total_budget}"
            )
        if total_budget <= initial_budget:
            raise ValueError(
                f"Error in budget processing; intial budget [{initial_budget}] greater than total budget [{total_budget}]"
            )

        self.spent_budget = initial_budget
        self.total_budget = total_budget

        if seed is not None : 
            random.seed(seed)
        # Build first set of indices based on the intial Budget

        self.stage3_indexes = [x for x in range(len(self.dataset))]
        assert len(self.stage2_indexes) + len(self.stage3_indexes) == len(dataset)

        self.stage2_transform = transform
        self.stage3_transform = transforms.Compose([transforms.ToTensor()])


    def is_stage2(self) -> bool:
        return self.current_stage == self.STAGE2

    def is_stage3(self) -> bool:
        return self.current_stage == self.STAGE3

    def goto_stage2(self) -> None:
        self.current_stage = self.STAGE2

    def goto_stage3(self) -> None:
        self.current_stage = self.STAGE3

    
    def query_oracle(self, indices: torch.LongTensor):

        indices = indices.tolist()

        top_scoring = np.asarray(self.stage3_indexes)[indices].tolist()
        self.stage2_indexes += top_scoring

        self.stage3_indexes = [
            x for x in self.stage3_indexes if x not in self.stage2_indexes
        ]
        self.stage2_indexes.sort()
        self.stage3_indexes.sort()

        self.spent_budget += len(indices) / len(self.dataset)
    
    def _sample_init(self, RotNet : nn.Module) : 

        RotNet.eval()
        
        # TODO: Remove hardcode cuda
        RotNet.to("cuda")


        _loader = DataLoader(self.dataset, 500, shuffle=False)
        
        _losses = list()

        for img, _ in _loader : 
            img.to("cuda")
            preds = RotNet(img)
            _losses.append(F.cross_entropy(preds, reduction="mean").tolist())





    def __len__(self):
        if self.current_stage == self.STAGE2:
            return len(self.stage2_indexes)
        else:
            return len(self.stage3_indexes)

    def __getitem__(self, index):
        if self.current_stage == self.STAGE2:
            images, labels = self.dataset[self.stage2_indexes[index]]
            images = self.stage2_transform(images)
            return images, labels
        else:
            images, labels = self.dataset[self.stage3_indexes[index]]
            images = self.stage3_transform(images)
            return images, labels





if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def plot_images(images) : 
        num_images = len(images)

        if num_images == 0 : 
            return 

        # images per row
        img_per_row = 2
        num_row = num_images // img_per_row
        fig, ax = plt.subplots(nrows=num_row, ncols=img_per_row)

        for i, axi in enumerate(ax.flat): 
            axi.imshow(images[i])
        
        plt.show()

        

        

    data = datasets.CIFAR10(root="data", train=True, download=False)

    ds = CIFAR10ActiveWrapper(
        data, 0.1, 0.5, 10, transform=transforms.ToTensor()
    )
    ds.goto_stage3()
    ds.query_oracle(torch.LongTensor([x for x in range(50)]))      

    loader = DataLoader(ds, batch_size=1)


    for i, (image,_) in enumerate(loader):
        # print(image)
        idx = ds.stage3_indexes[i]
        sample, _ = ds.dataset[idx]
        sample = transforms.ToTensor()(sample)
        sample = sample.permute(1,2,0).numpy()
        image = image[0].permute(1,2,0).numpy()
        if (sample != image).any() : 
            print("NOPE")
       
        plot_images([sample, image])    
