import torch 
import torchvision.datasets as datasets
from torch.utils.data import Dataset, Subset, IterableDataset, DataLoader
from torchvision import transforms

import random

class CIFAR10ActiveWrapper(Dataset) : 

    STAGE2 = 2
    STAGE3 = 3

    '''
    CIFAR10 wrapper for active learning
    '''
    def __init__(self, 
                 dataset        : datasets.CIFAR10, 
                 initial_budget : float, 
                 total_budget   : float , 
                 b              : int,  
                 initial_stage  : int = 2, 
                 transform      : transforms = None) :
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

        if initial_stage not in possible_stages : 
            raise ValueError(f"Error in stage: expected 2 or 3 but got {initial_stage}")

        if not isinstance(dataset, datasets.CIFAR10) : 
            raise TypeError(f"Error loading dataset: Expected type datasets.CIFAR10 but got {type(dataset)}")

        self.dataset = dataset
        self.current_stage = initial_stage

        # First assumption is that all of the data is unlabeled thus belongs to stage3 data
        self.index_map = {k : 3 for k in range(len(dataset))}

        self.stage2_indexes = list()
        self.stage3_indexes = list() 

        if total_budget >= 1 : 
            raise ValueError(f"Error in budget processing: Expected in range [0, 1] but got {total_budget}")
        if total_budget <= initial_budget: 
            raise ValueError(f"Error in budget processing; intial budget [{initial_budget}] greater than total budget [{total_budget}]")

        self.spent_budget = initial_budget
        self.total_budget = total_budget

        # Build first set of indices based on the intial Budget
        while len(self.stage2_indexes) != int(len(dataset) * initial_budget) : 
            idx = random.randint(0, len(dataset) - 1)

            if self.index_map[idx] == 3: 
                self.index_map[idx] = 2 
                self.stage2_indexes.append(idx)
        
        self.stage3_indexes = [x for x in self.index_map if self.index_map[x] == 3]

        self.stage2_data = Subset(self.dataset, self.stage2_indexes)
        self.stage3_data = Subset(self.dataset, self.stage3_indexes)

        assert(len(self.stage2_indexes) + len(self.stage3_indexes) == len(dataset))

        self.transform = transform
    

    def get_stage2_loader(self, batch_size, num_workers = 1) : 
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    def get_stage3_loader(self, batch_size, num_workers = 1) : 
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    def getLoaders(self, stage2_batch_size, stage3_batch_size, stage2_num_workers = 1, stage3_num_workers = 1): 
        return self.get_stage2_loader(stage2_batch_size, stage2_num_workers), self.get_stage3_loader(stage3_batch_size, stage3_num_workers)
    
    def is_stage2(self) -> bool : 
        return self.current_stage == self.STAGE2
    
    def is_stage3(self) -> bool : 
        return self.current_stage == self.STAGE3
    
    def goto_stage2(self) -> None: 
        self.current_stage = self.STAGE2
    
    def goto_stage3(self) -> None : 
        self.current_stage = self.STAGE3 
    
    
    def __len__(self) : 
        if self.current_stage == self.STAGE2: 
            return len(self.stage2_data)
        else : 
            return len(self.stage3_data)
    
    def __getitem__(self, index) :
        if self.current_stage == self.STAGE2: 
            images, labels = self.stage2_data[index]
            images = self.transform(images)
            return images, labels
        else: 
            images, _ = self.stage3_data[index]
            images = self.transform(images)
            return images



if __name__ == "__main__" : 
    data = datasets.CIFAR10(root = "data", train=True, download=False)
    ds = CIFAR10ActiveWrapper(data, .1, .5, 10, transform=transforms.ToTensor(), initial_stage=3)
    loader = DataLoader(ds, batch_size=1)

    for i, image in enumerate(loader): 
        # print(image)
        idx = ds.stage3_indexes[i]
        sample, _ = ds.stage3_data[idx]
        sample = transforms.ToTensor()(sample)
        print(sample == image)
        
        input()

