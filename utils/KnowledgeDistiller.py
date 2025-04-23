
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from logger import logPrint
from typing import List
import gc
from torch.nn.utils import parameters_to_vector
from torch import nn, optim, Tensor

from torch.nn import Module





class KnowledgeDistiller:
    """
    A class for Knowledge Distillation using ensembles.
    """

    def __init__(
        self,
        dataset,
        epoc=3,
        batch_size=16,
        temperature=2,  
        method="avglogits",
        device="cpu",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.T = temperature
        self.epoc = epoc
        self.lr1 = 0.001
        self.swa_lr = 0.005
       
      
        
        
        
        
        self.momentum = 0.9 #0.9 CIFAR10
        
        self.method = method 
        self.device = device
        self.optimizer_type = "SGD" # Adam"   could be "SGD", "Adagrad", etc.
        


        
        
    
    def distillKnowledge(self, ensemble, Scoresmodel):
        """
        Takes in a teacher ensemble (list of models) and a student model.
        Trains the student model using unlabelled dataset, then returns it.
        Args:
            teacher_ensemble is list of models used to construct pseudolabels using self.method
            student_model is models that will be trained
        """

        gc.collect()
        torch.cuda.empty_cache()
        # Set labels as soft ensemble prediction
        self.dataset.labels = self._pseudolabelsFromEnsemble(ensemble, self.method)
        print("LR:----" ,self.lr1)
        print("LRSWA:----", self.swa_lr)
        print("Epoch:----",self.epoc)
        print("optimizer_type:----",self.optimizer_type)

        if self.optimizer_type == "SGD":
            opt = optim.SGD(Scoresmodel.parameters(), momentum=self.momentum, lr=self.lr1, weight_decay=1e-5)
        
        Loss = nn.KLDivLoss
        loss = Loss(reduction="batchmean")
     

        swa_model = AveragedModel(Scoresmodel)
        #T_max_dynamic = self.epoc  # or any other dynamic determination
        #scheduler = CosineAnnealingLR(opt, T_max=T_max_dynamic)
        scheduler = CosineAnnealingLR(opt, T_max=100)
        swa_scheduler = SWALR(opt, swa_lr=self.swa_lr)

        dataLoader = DataLoader(self.dataset, batch_size=self.batch_size,)
        for i in range(self.epoc):
            
            total_err = 0
            for j, (x, y) in enumerate(dataLoader):

                
                opt.zero_grad()
                pred = Scoresmodel(x)


                err = loss(F.log_softmax(pred / self.T, dim=1), y) * self.T * self.T
                err.backward()
                total_err += err
                opt.step()
            logPrint(f"KD epoch {i}: {total_err}")
            scheduler.step()
            swa_model.update_parameters(Scoresmodel)
            swa_scheduler.step()

            torch.optim.swa_utils.update_bn(dataLoader, swa_model)


        gc.collect()
        torch.cuda.empty_cache()
        return swa_model.module
  

    
    
    
    
    def _pseudolabelsFromEnsemble(self, ensemble, method=None):
        """
        Combines the probabilities to make ensemble predictions.
        avglogits: Takes softmax of the average outputs of the models
           
        """
        if isinstance(ensemble, Module):
            # If it's a single model, convert it into a list
            ensemble = [ensemble]
        if method is None:
            method = self.method
        print (f" ensemble-------: {len(ensemble)}")
        print (f" Method-------: {len(method)}")
        

            
        with torch.no_grad():
            dataLoader = DataLoader(self.dataset, batch_size=self.batch_size)
            preds = []
            for i, (x, y) in enumerate(dataLoader):
       
                predsBatch = torch.stack([m(x) / self.T for m in ensemble])
                preds.append(predsBatch)
            preds = torch.cat(preds, dim=1)
            print(f"Final preds shape: {preds.shape}")


            
            
            if method == "avglogits":
                pseudolabels = preds.mean(dim=0)
                return F.softmax(pseudolabels, dim=1)
            
            
            else:
                raise ValueError(
                    "pseudolabel method should be one of: avglogits, medlogits, avgprob"
                )
                

    


    
    
   