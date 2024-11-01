#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
#import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader



transform = transforms.Compose([transforms.ToTensor()])

batch_size = 4

class CustomDataset(Dataset):
 def __init__(self, csv_file_path):
 # Initialize the dataset with path a csv file containing the data. I have ad
    data = pd.read_csv(csv_file_path, sep=' ')

    self.labels = data['label']
    self.features1 = data['feature1']
    self.features2 = data['feature2']

 def __len__(self):
    # returning length
    return len(self.labels)
 def __getitem__(self, idx):
    feature1 = self.features1[idx]
    feature2 = self.features2[idx]
    label = self.labels[idx]
    return label, feature1, feature2

# Set up the dataset.
dataset = r'C:\1Masters\DeepLearning\trainData.txt'
training_dataset = CustomDataset(dataset)



# Set up the dataset.
trainloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)

# get some images
dataiter = iter(trainloader)
images, labels = next(dataiter)


for i in range(5): #Run through 5 batches
    images, labels = next(dataiter)
    for image, label in zip(images,labels): # Run through all samples in a batch
        plt.figure()
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        plt.title(label)
