import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms

class Classifier(nn.Module):
    def __init__(self, num_classes= 525):
        super(Classifier, self).__init__()
        
        # Load a pre-trained ResNet50 model
        self.model = models.resnet50(pretrained=True)

        # Freeze earlier layers if you want to fine-tune only the later layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last few layers for fine-tuning
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        
        # Replace the fully connected layer to match the number of bird species (400 classes)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),  # Fully connected layer with 1024 units
            nn.BatchNorm1d(1024),       # Add batch normalization
            nn.ReLU(),                  # ReLU activation function
            nn.Dropout(0.5),            # Dropout for regularization (50% dropout)
            nn.Linear(1024, num_classes) # Final layer with output size = number of bird species
        )
        
    def forward(self, x):
        return self.model(x)