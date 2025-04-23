from typing import List, Optional, Tuple
import os  
from torchvision.io import read_image
import logging

import numpy as np
from datasetLoaders.DatasetLoader import DatasetLoader
from datasetLoaders.DatasetInterface import DatasetInterface
from pandas import DataFrame
from torch import Tensor
from torchvision import transforms, datasets
from logger import logPrint
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, ColorJitter
from PIL import Image

from torch.utils.data import DataLoader
import pandas as pd
import torch

import os
import logging
import torch
from PIL import Image
from torchvision import transforms, datasets
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, ColorJitter
from torch.utils.data import DataLoader, TensorDataset, Dataset, WeightedRandomSampler
from typing import List, Tuple, Optional
import pandas as pd
from pandas import DataFrame

class DatasetLoaderCIFAR10(DatasetLoader):

    def __init__(self, synthetic_data_path='./data/cifar'):
        self.synthetic_data_path = synthetic_data_path
        super().__init__()

    def logPrint(self, message: str):
        print(message)

    def getDatasets(
        self,
        percUsers: torch.Tensor,
        labels: torch.Tensor,
        size: Optional[Tuple[int, int]] = None,
        nonIID=True,
        alpha=0.7,
        percServerData=0.15
    ) -> Tuple[List[DatasetInterface], DatasetInterface, DatasetInterface]:
        
        self.logPrint("Loading CIFAR10...")
        self._setRandomSeeds()
        trainDataframe, testDataframe = self.__loadCIFAR10Data()

        self.logPrint(f"Loaded CIFAR10 train data size: {len(trainDataframe)}, test data size: {len(testDataframe)}")

        # Load synthetic data for the server
        syntheticDataframe = self.__loadSyntheticData()
        if syntheticDataframe is None:
            raise ValueError("Failed to load synthetic data")

        self.logPrint(f"Loaded synthetic data size: {len(syntheticDataframe)}")

        # Ensure the server dataset has exactly 14,523 samples
        if len(syntheticDataframe) > 14523:
            syntheticDataframe = syntheticDataframe.sample(n=14523, random_state=1).reset_index(drop=True)
        elif len(syntheticDataframe) < 14523:
            raise ValueError("Synthetic data contains fewer than 14,523 samples")

        # Initialize the server dataset with synthetic data
        serverDataset = self.CIFAR10Dataset(syntheticDataframe.reset_index(drop=True))
        self.logPrint(f"Server dataset size: {len(serverDataset)}")
        self.logPrint(f"Lengths of server {len(serverDataset)} and train {len(trainDataframe)}")

        # Save some images from the synthetic and training datasets
        #self.save_visualized_images(syntheticDataframe, trainDataframe)

        # Process training and test data
        clientDatasets = self._splitTrainDataIntoClientDatasets(percUsers, trainDataframe, self.CIFAR10Dataset, nonIID, alpha)
        testDataset = self.CIFAR10Dataset(testDataframe)
        
        return clientDatasets, testDataset, serverDataset

    def save_visualized_images(self, syntheticDataframe: pd.DataFrame, trainDataframe: pd.DataFrame, num_images: int = 5, save_folder: str = './visualized_images'):
        """
        Saves a few images from the synthetic and training datasets to ensure they are different.
        """
        def save_images(dataframe, folder, prefix):
            os.makedirs(folder, exist_ok=True)
            for i in range(num_images):
                img = torch.tensor(dataframe.iloc[i]['data'])
                img = img.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
                img = img * torch.tensor([0.2023, 0.1994, 0.2010]) + torch.tensor([0.4914, 0.4822, 0.4465])  # De-normalize
                img = img.numpy()
                img = (img * 255).astype(np.uint8)  # Convert to uint8
                pil_img = Image.fromarray(img)
                label = dataframe.iloc[i]['labels']
                img_path = os.path.join(folder, f'{prefix}_img_{i}_label_{label}.png')
                pil_img.save(img_path)
                print(f"Saved image: {img_path}")

        synthetic_folder = os.path.join(save_folder, 'synthetic')
        train_folder = os.path.join(save_folder, 'train')

        self.logPrint("Saving synthetic dataset images...")
        save_images(syntheticDataframe, synthetic_folder, 'synthetic')

        self.logPrint("Saving training dataset images...")
        save_images(trainDataframe, train_folder, 'train')

    def __loadSyntheticData(self) -> pd.DataFrame:
        categories = ['truck', 'ship', 'horse', 'frog', 'dog', 'deer', 'cat', 'bird', 'automobile', 'airplane']
        imageData, imageLabels = [], []

        syntheticTransform = Compose([
            Resize((32, 32)),  # Adjust size as needed.
            ToTensor(),  # Convert to tensor and scale pixel values to [0, 1]
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        for idx, category in enumerate(categories):
            folderPath = os.path.join(self.synthetic_data_path, category)
            if not os.path.exists(folderPath):
                self.logPrint(f"Category folder not found: {folderPath}")
                continue
            for imageName in os.listdir(folderPath):
                imagePath = os.path.join(folderPath, imageName)
                if not os.path.isfile(imagePath):
                    self.logPrint(f"Image file not found: {imagePath}")
                    continue
                try:
                    image = read_image(imagePath)
                    image = Image.fromarray(image.permute(1, 2, 0).numpy())  # Convert tensor to PIL image
                    image = syntheticTransform(image)
                    imageData.append(image.numpy())
                    imageLabels.append(idx)
                except Exception as e:
                    logging.error(f"Error loading image {imagePath}: {e}")
                    continue

        if not imageData or not imageLabels:
            self.logPrint("No synthetic data loaded")
            return None

        syntheticDataframe = pd.DataFrame({'data': imageData, 'labels': imageLabels})
        return syntheticDataframe

    def __loadCIFAR10Data(self) -> Tuple[DataFrame, DataFrame]:
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainSet = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testSet = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        trainLoader = DataLoader(trainSet, batch_size=len(trainSet), shuffle=False)
        testLoader = DataLoader(testSet, batch_size=len(testSet), shuffle=False)

        xTrain, yTrain = next(iter(trainLoader))
        xTest, yTest = next(iter(testLoader))

        trainDataframe = DataFrame({'data': list(xTrain.numpy()), 'labels': yTrain.numpy()})
        testDataframe = DataFrame({'data': list(xTest.numpy()), 'labels': yTest.numpy()})

        return trainDataframe, testDataframe

    class CIFAR10Dataset(Dataset):
        def __init__(self, dataframe):
            """
            Initializes the dataset with data and labels.
            Assumes that dataframe['data'] contains numpy arrays or tensors.
            """
            # Checking if the data in the dataframe is numpy arrays and converting them to tensors
            self.data = torch.stack(
                  [torch.from_numpy(data) if isinstance(data, np.ndarray) else data
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