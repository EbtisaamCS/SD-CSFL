from utils.typings import Errors, Accuracy, AttackSuccesRate, attack_success_variance
from experiment.AggregatorConfig import AggregatorConfig
from torch import nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from client import Client
from logger import logPrint
from typing import List, Tuple
import numpy as np
import os
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from datasetLoaders.CIFAR10 import DatasetLoaderCIFAR10
from datasetLoaders.CIFAR100 import DatasetLoaderCIFAR100
from sklearn.ensemble import IsolationForest
import torch
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.normal import Normal
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from utils.KnowledgeDistiller import KnowledgeDistiller
from scipy.stats import zscore
from sklearn.cluster import KMeans
from torch.nn.utils import parameters_to_vector
from sklearn.metrics.pairwise import cosine_similarity
import copy
from torch.nn import Module
import gc
import json
from sklearn.decomposition import PCA as pca_func
import hdbscan
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import trim_mean
import random
from scipy.stats import norm
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from PIL import Image
from datasetLoaders.Birds import DatasetLoaderBirds


class SCRFAAggregator(Aggregator):
    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
    ):  
        
        super().__init__(clients, model, config, useAsyncClients)

        logPrint("INITIALISING Aggregator!")
        # Unlabelled data which will be used in Knowledge Distillation
        self.distillationData = None  # data is loaded in __runExperiment function
        self.sampleSize = config.sampleSize
        self.true_labels = None
        self.pseudolabelMethod = "avglogits"  #"for 3Mluti avglogits" #"medlogits" avglogits
        print(f"distillationData Data1: {self.distillationData}")

        self._debug_distillation_data()

    def _debug_distillation_data(self):
        """
        Method to print the structure and some content of distillationData for debugging.
        """
        print("Debugging distillationData:")
        if isinstance(self.distillationData, TensorDataset):
            data_tensors = self.distillationData.tensors
            print(f"Number of tensors: {len(data_tensors)}")
            for i, tensor in enumerate(data_tensors):
                print(f"Tensor {i} shape: {tensor.shape}")
                print(f"Tensor {i} content (first 5 elements): {tensor[:5]}")
        elif isinstance(self.distillationData, DatasetLoaderCIFAR10.CIFAR10Dataset):
            print(f"Number of samples: {len(self.distillationData)}")
            print(f"Data sample shape: {self.distillationData[0][0].shape}")
            print(f"Label sample: {self.distillationData[0][1]}")
        else:
            print("distillationData is not a recognized dataset type")

            
    def trainAndTest(self, testDataset: DatasetInterface) ->Tuple[Errors, AttackSuccesRate]:
        roundsError = torch.zeros(self.rounds)
        roundsattack_success_rate = torch.zeros(self.rounds)
        
        kd = KnowledgeDistiller(
             self.distillationData)
        # Ensure distillationData is not None
        print(f"Calibration set created from synthetic data11: {self.distillationData}")

        assert self.distillationData is not None, "distillationData must be set before calling trainAndTest."

        print("Verifying distillationData before creating calibration set:")
        self._debug_distillation_data()
        # Creating DataLoader from the synthetic data
        if isinstance(self.distillationData, DatasetLoaderCIFAR10.CIFAR10Dataset):
            print(f"Calibration set created from synthetic data12: {self.distillationData}")
            
            
        else:
            raise ValueError("distillationData must be an instance of birds or cifar10")

        calibration_set = DataLoader(self.distillationData, batch_size=16,shuffle=True)
        print(f"Calibration set created from synthetic data: {self.distillationData}")
        print(f"Calibration set created from synthetic data1: {calibration_set}")
        total_samples = len(self.distillationData)
        print(f"Total number of samples in the calibration set: {total_samples}")
        num_batches = len(calibration_set)
        print(f"Total number of batches in the calibration set: {num_batches}")


        for self.r in range(self.rounds):

            print(f"Round... {self.r}")

            if self.r == 0:
                self.Globalmodel0 = self._shareModelAndTrainOnClients()
                self.prevmodelparm = [self.flatten_parameters(self.Globalmodel0).cpu().detach().numpy()]
                print(f"Global model0 ---: {self.prevmodelparm}")
            else:
                for client in self.chosen_clients1:
                    self.__shareModelAndTrainOnClient(client, model=self.model)
                

                # Handle malicious clients separately
                #malicious_clients = [self.clients[i] for i in self.malicious_indices]

                # Create a perturbed version of the aggregation model for malicious clients
                #perturbed_model = copy.deepcopy(self.model)
                #for param in perturbed_model.parameters():
                    #noise = torch.randn_like(param) * 1e-4  # Minimal perturbation
                    #param.data += noise

                # Share the perturbed model with malicious clients
                #for client in malicious_clients:
                    #self.__shareModelAndTrainOnClient(client, model=perturbed_model)


            self.models = self._retrieveClientModelsDict()
            self.modelsparm = [self.flatten_parameters(model).cpu().detach().numpy() for model in self.models]
            #print(f"Retrieved client models ---: {self.modelsparm}")

            self.filtered_models = self.identify_and_exclude_malicious_clients(self.models, calibration_set)

            self.chosen_clients1 = [self.clients[i] for i in self.chosen_indices if i in self.benign_indices]
            print(f"Chosen benign clients for aggregation: {self.chosen_clients1}")

            self.model = self.aggregate(self.chosen_clients1, self.filtered_models)
            self.prevmodel = self.model

            error, success_rate = self.test(testDataset)
            roundsError[self.r] = torch.tensor(error, dtype=torch.float32)
            roundsattack_success_rate[self.r] = torch.tensor(success_rate, dtype=torch.float32)

            print(f'Accuracy of the model on clean images: {roundsError[self.r]:.2f}%')
            print(f'Attack success rate: {roundsattack_success_rate[self.r]:.2f}%')

        return roundsError, roundsattack_success_rate

    def aggregate(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:
        avg_model = self._averageModel(models)
        return avg_model
    
    


    
    def flatten_parameters(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)
    
    def identify_and_exclude_malicious_clients(self, models: List[torch.nn.Module], calibration_set: DataLoader) -> List[torch.nn.Module]:
        nonconformity_scores = []
        nonconformity_scores = [self.calculate_nonconformity_score(model, calibration_set) for model in models]
        
        # Classify clients using the refined threshold method
        self.benign_indices, self.malicious_indices = self.classify_clients(nonconformity_scores)

        print(f"Nonconformity scores: {nonconformity_scores}")
        print(f"Benign client indices: {self.benign_indices}")
        print(f"Malicious client indices: {self.malicious_indices}")

        # Filter models based on benign indices
        filtered_models = [models[i] for i in self.benign_indices]
        return filtered_models

    
    def calculate_nonconformity_score(self, model, calibration_set: DataLoader) -> float:
        model.eval()
        entropy_scores = []
        
        # Create a balanced calibration set to address non-IID data
        balanced_calibration_set = self.create_balanced_calibration_set(calibration_set)
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.visualize_balanced_calibration_set(calibration_set, device=device)

        with torch.no_grad():
            for data, target in balanced_calibration_set:
            #for data, target in calibration_set:

                if data.dim() == 3:
                    data = data.unsqueeze(0)  # Handle single image case
                output = model(data)
                probs = F.softmax(output, dim=1)
                probs = torch.clamp(probs, min=1e-9)  # Prevent log(0)
                entropy = -(probs * torch.log(probs)).sum(1).mean()  # Measure of uncertainty
                entropy_scores.append(entropy.cpu().item() if not torch.isnan(entropy) else 0)  # Replace nan with 0 or other neutral value
        
        entropy_mean = np.nanmean(entropy_scores)  # Use nanmean to ignore nan values in mean calculation
        return entropy_mean


    


    


    def create_balanced_calibration_set(self, calibration_set: DataLoader) -> DataLoader:
        # Create a stratified sampler to balance the calibration set
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        targets = []
        for _, target in calibration_set:
            targets.extend(target.cpu().numpy().flatten())  # Move to CPU and convert to numpy, then flatten
        targets = np.array(targets)
        
        class_sample_counts = np.bincount(targets)
        weights = 1. / class_sample_counts
        samples_weights = weights[targets]
        sampler = torch.utils.data.WeightedRandomSampler(samples_weights, len(samples_weights))
        
        balanced_calibration_loader = DataLoader(calibration_set.dataset, batch_size=calibration_set.batch_size, sampler=sampler)

        
        return balanced_calibration_loader

    def classify_clients(self, nonconformity_scores):
        # Define more narrowly focused thresholds
        low_thresh = 27 #22 60 0r 70 (IPM) 10 60 (ALiE) --birds IPM ALiE 25 55 f3ba 20 50 a3fl 45 90 13 60 32 55 Crep 27 60 , 29 57
        high_thresh = 60



        print(f"Low Threshold-------: {low_thresh}")
        print(f"High Threshold------: {high_thresh}")

        low_threshold = np.percentile(nonconformity_scores, low_thresh)
        high_threshold = np.percentile(nonconformity_scores, high_thresh)

        print(f"Low Threshold: {low_threshold}")
        print(f"High Threshold: {high_threshold}")

        # Classifying clients based on thresholds
        benign_indices = [i for i, score in enumerate(nonconformity_scores) if low_threshold <= score <= high_threshold]
        malicious_indices = [i for i, score in enumerate(nonconformity_scores) if score < low_threshold or score > high_threshold]
        return benign_indices, malicious_indices
    
    
       

    def __shareModelAndTrainOnClient(self, client: Client, model: nn.Module):
        broadcastModel = copy.deepcopy(model)
        client.updateModel(broadcastModel)
        error, pred = client.trainModel()

    @staticmethod
    def requiresData():
        """
        Returns boolean value depending on whether the aggregation method requires server data or not.
        """
        return True   



