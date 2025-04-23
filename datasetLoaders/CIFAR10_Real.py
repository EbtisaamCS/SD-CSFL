from typing import List, Optional, Tuple

import numpy as np
from datasetLoaders.DatasetLoader import DatasetLoader
from datasetLoaders.DatasetInterface import DatasetInterface
from pandas import DataFrame
import pandas as pd  # Ensure pandas is imported

from torch import Tensor
from torchvision import transforms, datasets
from logger import logPrint
import torch
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, ColorJitter

class DatasetLoaderCIFAR10(DatasetLoader):


    def getDatasets(
        self,
        percUsers: Tensor,
        labels: Tensor,
        size: Optional[Tuple[int, int]] = None,
        nonIID=True,
        alpha=0.7,
        percServerData=0.15,) -> Tuple[List[DatasetInterface], DatasetInterface]:
        logPrint("Loading CIFR10...")
        self._setRandomSeeds()
        data = self. __loadCIFAR10Data()  # Load CIFAR10 data instead of MNIST
        trainDataframe, testDataframe = self._filterDataByLabel(labels, *data)
        serverDataset = []
        if percServerData > 0:
            msk = np.random.rand(len(trainDataframe)) < percServerData
            serverDataframe, trainDataframe = trainDataframe[msk], trainDataframe[~msk]
            serverDataset = self.CIFAR10Dataset(serverDataframe.reset_index(drop=True))  
            logPrint(f"Lengths of server {len(serverDataframe)} and train {len(trainDataframe)}")
        else:
            logPrint(f"Lengths of server {0} and train {len(trainDataframe)}")
        clientDatasets = self._splitTrainDataIntoClientDatasets(percUsers, trainDataframe, self.CIFAR10Dataset, nonIID, alpha)
        testDataset = self.CIFAR10Dataset(testDataframe)  # Use CIFAR10DAtaset
        return clientDatasets, testDataset, serverDataset
    

           
    @staticmethod
    def __loadCIFAR10Data() -> Tuple[DataFrame, DataFrame]:
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
        trainSet = datasets.CIFAR10(root='./data', train=True, download=True, transform= transform_train)
        testSet = datasets.CIFAR10(root='./data', train=False, download=True, transform= transform)
        
    
        xTrain, yTrain = zip(*[(x, y) for x, y in DataLoader(trainSet)])
        xTest, yTest = zip(*[(x, y) for x, y in DataLoader(testSet)])
    
        xTrain, yTrain = torch.cat(xTrain), torch.tensor(yTrain)
        xTest, yTest = torch.cat(xTest), torch.tensor(yTest)
    
        trainDataframe = DataFrame({'data': [x for x in xTrain], 'labels': yTrain.numpy()})
        testDataframe = DataFrame({'data': [x for x in xTest], 'labels': yTest.numpy()})
    
        return trainDataframe, testDataframe

    class CIFAR10Dataset(DatasetInterface):
        def __init__(self, dataframe):
            """
            Initializes the dataset with data and labels.
            Assumes that dataframe['data'] contains numpy arrays or tensors.
            """
            # Checking if the data in the dataframe is numpy arrays and converting them to tensors
            self.data = torch.stack(
                  [data if isinstance(data, torch.Tensor) else torch.from_numpy(data) 
                   for data in dataframe["data"].values])
            
            self.labels = torch.tensor(dataframe["labels"].values)
          
        def get_labels(self) -> torch.Tensor:
            """
            Returns the labels as a tensor.
            """
            return self.labels

        def __len__(self):
            """
            Returns the number of samples in the dataset.
            """
            return len(self.data)

        def __getitem__(self, index):
            """
            Returns the data and label corresponding to the index.
            """
            return self.data[index], self.labels[index]

        def to(self, device):
            """
            Moves the data and labels to the specified device.
            """
            self.data = self.data.to(device)
            self.labels = self.labels.to(device)

