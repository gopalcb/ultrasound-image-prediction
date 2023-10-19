# Importing libraries
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os
from PIL import Image
import torch.optim as optim
from matplotlib import pyplot as plt

from training.batch import *
from training.fccn import Net


# Hyper-parameter settings
TOTAL_EPOCH = 200
LEARNING_RATE = 0.001
MOMENTUM = 0.9
BATCH_SIZE = 5
class_arr = [0,1]
MODEL_NAME_POSTFIX = '-local-on-gpu'


# Set training device: CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)


def train_model():
    # Training data on GPU
    train_loss = []
    train_accuracy = []

    test_loss = []
    test_accuracy = []

    # initialize fccn neural net
    net = Net()

    print('Training started...\n')
    batch_features, batch_classes, classes = generate_training_batch()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=0.001) #, weight_decay=0.001

    for epoch in range(TOTAL_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        end_loss = 0.0
        j = 0
        
        y_hat_classes = []
        
        for i in range(0, len(batch_features)):
            
            print('\rEPOCH ' + str(epoch+1) + '/'+str(TOTAL_EPOCH)+': Batch ' + str(i+1) + '/' + str(len(batch_features)), end='')
            
            inputs = batch_features[i].to(device)
            labels = batch_classes[i].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #print(outputs)
            
            # print statistics
            running_loss += loss.item()
            end_loss= loss.item()
            
            j += 1
            
            #y_hat_class = np.where(outputs.detach().cpu().numpy()<0.5, 0, 1)
            _, predicted = torch.max(outputs, 1)
            #print(predicted)
            y_hat_class = predicted.cpu().numpy()
            
            #y_hat_classes.append(y_hat_class[0][1])
            #y_hat_classes.append(y_hat_class[1][1])
            y_hat_classes.append(y_hat_class)
            
        avg_loss = running_loss/j
        
        y_hat_classes = np.array(y_hat_classes)
        accuracy = np.sum(classes.reshape(-1,1)==y_hat_classes.reshape(-1,1)) / len(classes.reshape(-1,1))
        
        train_accuracy.append(accuracy)
        train_loss.append(avg_loss)

        print('\nEPOCH ' + str(epoch+1) + '/'+str(TOTAL_EPOCH)+'(DONE): ' + 'Loss = ' + str(avg_loss) + 
            '  Accuracy = ' + str(accuracy) + 
            ' (' + 'GPU: cuda:0 - ' + torch.cuda.get_device_name(0) + ')\n')

    # Save trained model
    PATH = f'D:/SCA/256_256/nn_output/trained-model-sca.net'
    torch.save(net, PATH)

    print('\nTraining complete')
    return train_loss, train_accuracy
