# Fully Connected Neural Network (FCCN)
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.classifier = nn.Sequential(
            
            nn.Linear(120, 50),
            nn.BatchNorm1d(50),
            nn.Dropout(0.30),
            nn.ReLU(),

            nn.Linear(50, 50),
            nn.BatchNorm1d(50),
            nn.Dropout(0.30),
            nn.ReLU(),
            
            nn.Linear(50, 40),
            nn.BatchNorm1d(50),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(40, 30),
            nn.BatchNorm1d(40),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Linear(30, 15),
            nn.BatchNorm1d(15),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Linear(15, 10),
            nn.BatchNorm1d(15),
            nn.Dropout(0.1),
            nn.ReLU(),
            
            nn.Linear(10, 2),
            #nn.Softmax(dim=1)
            nn.Sigmoid()
        )

    
    def forward(self, x):
        x = x.view(-1, 120)
        
        x = self.classifier(x)
        
        return x