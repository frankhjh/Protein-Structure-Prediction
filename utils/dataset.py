#!/usr/bin/env python
from torch.utils.data import Dataset

class trainset(Dataset):
    def __init__(self,proteins,labels):
        super(trainset,self).__init__()
        self.proteins=proteins
        self.labels=labels
    
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self,idx):
        return self.proteins[idx],self.labels[idx]
    
class testset(Dataset):
    def __init__(self,proteins):
        super(testset,self).__init__()
        self.proteins=proteins
    
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self,idx):
        return self.proteins[idx]
        