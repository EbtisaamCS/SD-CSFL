from utils.typings import Errors, Accuracy, AttackSuccesRate
from experiment.AggregatorConfig import AggregatorConfig
from torch import nn
from client import Client
from logger import logPrint
from typing import List,Tuple
import torch
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface
from datasetLoaders.DatasetLoader import DatasetLoader
from torch import Tensor
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from scipy.ndimage import rotate, zoom, shift
import cv2
import random

from copy import deepcopy
import gc

class MedianAggregator(Aggregator):
    """
    COrdinated MEDian (COMED) Aggregator.

    Uses the median parameter across all the models parameters for each parameter.
    """

    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients=False,
    ):
        super().__init__(clients, model, config, useAsyncClients)

    def trainAndTest(self, testDataset: DatasetInterface) -> Tuple[torch.Tensor, torch.Tensor]:
        roundsattack_success_rate = torch.zeros(self.rounds)
        roundsError = Errors(torch.zeros(self.rounds))

        for r in range(self.rounds):
            logPrint("Round... ", r)

            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()

            # Merge models
            chosen_clients = [self.clients[i] for i in self.chosen_indices]
            self.model = self.aggregate(chosen_clients, models)

            # Compute error and attack success rate
            error, success_rate = self.test(testDataset)
            roundsError[r] = torch.tensor(error, dtype=torch.float32)
            roundsattack_success_rate[r] = torch.tensor(success_rate, dtype=torch.float32)
            #attack_success_variance[r] = torch.tensor(variance, dtype=torch.float32)

            print(f'Accuracy of the model on clean images: {roundsError[r]:.2f}%')
            print(f'Attack success rate: {roundsattack_success_rate[r]:.2f}%')
            #print(f'attack_success_variance: {attack_success_variance[r]:.2f}%')

        return roundsError, roundsattack_success_rate


    def aggregate(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:
        # We can't do aggregation if there are no models this round
        if (len(models)) == 0:
            return self.model.to(self.device)

        model = models[0]
        modelCopy = deepcopy(model)

        for name1, _ in model.named_parameters():
            m = []
            for i in range(len(clients)):
                params2 = models[i].named_parameters()
                dictParams2 = dict(params2)
                m.append(dictParams2[name1].data.view(-1).to("cpu").numpy())

            m = torch.tensor(m)
            med = torch.median(m, dim=0)[0]
            dictParamsm = dict(modelCopy.named_parameters())
            dictParamsm[name1].data.copy_(med.view(dictParamsm[name1].data.size()))

        return modelCopy.to(self.device)
