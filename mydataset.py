#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:26:11 2020

@author: naveenpaluru
"""



import torch
from torch.utils.data import Dataset



class myDataset(Dataset):
    def __init__(self, images, labels, transforms):
        self.X = images
        self.Y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        datax = self.X[i, :]
        datay = self.Y[i, :]                     
        if self.transforms:
            datax = self.transforms(datax).float()  
            datay = torch.LongTensor(datay)            
        return datax, datay
 
