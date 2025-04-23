import os
import pickle
import logging
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pandas as pd
from typing import List, Tuple, Optional
from datasetLoaders.DatasetLoader import DatasetLoader
from datasetLoaders.DatasetInterface import DatasetInterface
import logging


from typing import List, Tuple, Optional
import numpy as np
from sklearn.utils import resample

class DatasetLoaderBirds(DatasetLoader):
    def __init__(self, data_path='./data/Birds525/train', test_data_path='./data/Birds525/test', synthetic_data_path='./data/birds-2', 
                 num_categories=525, total_train_samples=50000, total_synthetic_samples=20850):
        self.data_path = data_path
        self.test_data_path = test_data_path
        self.synthetic_data_path = synthetic_data_path
        self.num_categories = num_categories
        self.total_train_samples = total_train_samples
        self.total_synthetic_samples = total_synthetic_samples
        self.cache_dir = './cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        super().__init__()

    def logPrint(self, message: str):
        logging.info(message)

    def getDatasets(
        self,
        percUsers: torch.Tensor,
        labels: torch.Tensor,
        size: Optional[Tuple[int, int]] = None,
        nonIID=False,
        alpha=0.9,
        percServerData=0.15
    ) -> Tuple[List[Dataset], Dataset, Dataset]:
        self.logPrint("Loading Birds dataset...")

        train_data_cache_path = os.path.join(self.cache_dir, 'train_data.pkl')
        test_data_cache_path = os.path.join(self.cache_dir, 'test_data.pkl')
        synthetic_data_cache_path = os.path.join(self.cache_dir, 'synthetic_data.pkl')

        if os.path.exists(train_data_cache_path) and os.path.exists(test_data_cache_path):
            self.logPrint("Loading cached training and test data...")
            with open(train_data_cache_path, 'rb') as f:
                train_dataframe = pickle.load(f)
            with open(test_data_cache_path, 'rb') as f:
                test_dataframe = pickle.load(f)
        else:
            self.logPrint("Loading and caching training and test data...")
            train_dataframe, test_dataframe = self.__loadTrainingData()
            with open(train_data_cache_path, 'wb') as f:
                pickle.dump(train_dataframe, f)
            with open(test_data_cache_path, 'wb') as f:
                pickle.dump(test_dataframe, f)

        self.logPrint(f"Loaded Birds train data size: {len(train_dataframe)}, test data size: {len(test_dataframe)}")

        if os.path.exists(synthetic_data_cache_path):
            self.logPrint("Loading cached synthetic data...")
            with open(synthetic_data_cache_path, 'rb') as f:
                synthetic_dataframe = pickle.load(f)
        else:
            self.logPrint("Loading and caching synthetic data...")
            synthetic_dataframe = self.__loadSyntheticData()
            with open(synthetic_data_cache_path, 'wb') as f:
                pickle.dump(synthetic_dataframe, f)

        self.logPrint(f"Loaded synthetic data size: {len(synthetic_dataframe)}")

        serverDataset = self.BirdsDataset(synthetic_dataframe.reset_index(drop=True))
        self.logPrint(f"Server dataset size: {len(serverDataset)}")

        clientDatasets = self._splitTrainDataIntoClientDatasets(percUsers, train_dataframe, self.BirdsDataset, nonIID, alpha)
        testDataset = self.BirdsDataset(test_dataframe)
        
        return clientDatasets, testDataset, serverDataset

    def __loadSyntheticData(self) -> pd.DataFrame:
        return self.__loadData(self.synthetic_data_path, max_samples=self.total_synthetic_samples)

    def __loadTrainingData(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_dataframe = self.__loadData(self.data_path, max_samples=self.total_train_samples)
        test_dataframe = self.__loadData(self.test_data_path)
        return train_dataframe, test_dataframe

    def __loadData(self, path, max_samples=None) -> pd.DataFrame:
        categories = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])[:self.num_categories]
        imageData = []
        imageLabels = []
        category_data = {}
        samples_loaded = 0

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        for idx, category in enumerate(categories):
            folderPath = os.path.join(path, category)
            category_images = os.listdir(folderPath)
            category_images_data = []
            
            for imageName in category_images:
                imagePath = os.path.join(folderPath, imageName)
                if imagePath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    try:
                        image = Image.open(imagePath).convert('RGB')
                        image = transform(image)
                        category_images_data.append((image.numpy(), idx))
                    except Exception as e:
                        self.logPrint(f"Error loading image {imagePath}: {e}")

            category_data[idx] = category_images_data

        imageData, imageLabels = self._balance_categories(category_data, max_samples)

        if not imageData:
            return pd.DataFrame()

        return pd.DataFrame({'data': imageData, 'labels': imageLabels})

    def _balance_categories(self, category_data, max_samples):
        if max_samples is None:
            max_samples = sum(len(images) for images in category_data.values())
        
        samples_per_category = max_samples // len(category_data) if max_samples else None
        balanced_data = []
        balanced_labels = []
        for category, images in category_data.items():
            if samples_per_category is None or len(images) <= samples_per_category:
                resampled_images = resample(images, replace=True, n_samples=samples_per_category or len(images), random_state=0)
            else:
                resampled_images = resample(images, replace=False, n_samples=samples_per_category, random_state=0)

            balanced_data.extend([img[0] for img in resampled_images])
            balanced_labels.extend([img[1] for img in resampled_images])

        return balanced_data, balanced_labels

    class BirdsDataset(Dataset):
        def __init__(self, dataframe):
            self.data = torch.stack(
                [torch.tensor(img, dtype=torch.float32) for img in dataframe['data'].values]
            )
            self.labels = torch.tensor(dataframe['labels'].values, dtype=torch.long)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

        def get_labels(self):
            return self.labels
