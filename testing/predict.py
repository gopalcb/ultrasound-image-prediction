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

from training.fccn import Net
from testing.data_loader import *


# Set training device: CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)


def predict():
    model = Net()
    model.load_state_dict(torch.load('D:/SCA/256_256/nn_output/trained-model-sca.net'))

    test_feature_map, test_classes, test_files_title = load_test_dataset()
    
    # get prediction on test data
    outputs = model(test_feature_map.to(device))
    print(outputs)
    return outputs


def compute_accuracy(model_outputs):
    res = []
    pred_accuracy_dict = []
    t_out = model_outputs

    # manually change threshold to test with different values
    threshold=0.6

    test_feature_map, test_classes, test_files_title = load_test_dataset()

    op = 0
    print('Computing accuracy...')
    for i in range(0, len(t_out)):
        #print(t_out)
        #op = c/100
        pred_val = t_out[i][0]
        #print(str(v))
        if pred_val > threshold:
            res.append(1)

            # t = title, c = catheter
            # f = file
            # a = accuracy
            d = {'t': 'c', 'f': test_files_title[i].split('.')[0], 'a': t_out[i][1]}
            pred_accuracy_dict.append(d)

        else:
            res.append(0)

            # t = title, c = catheter
            # f = file
            # a = accuracy
            d = {'t': 'e', 'f': test_files_title[i].split('.')[0], 'a': t_out[i][0]}
            pred_accuracy_dict.append(d)

    res = np.array(res)        
    predicted = torch.from_numpy(res)

    nd_predict = predicted.cpu().numpy()
    #####
    correct = 0
    total = 0

    with torch.no_grad():
        total = test_classes.size(0)
        correct = (nd_predict == test_classes.numpy()).sum().item()

    acc = (correct / total)

    acu = "%.4f" % acc  # computed accuracy
    print('Accuracy on '+ str(total) +' testset: ' + str(acu))

    return acc



def compute_combined_accuracy(pred_accuracy_dict, unet_prediction_accuracies, test_classes, t_out):
    combined_predict = []
    comb_pred_accuracy_dict = []

    for i in range(0, len(t_out)):
        unet_dict = unet_prediction_accuracies[i]
        native_dict = pred_accuracy_dict[i]
        
        new_accuracy = (float(unet_dict.get('a'))+float(native_dict.get('a')))/2
            
        if new_accuracy > 0.7:
            combined_predict.append(1)

            d = {'t': 'c', 'f': unet_dict.get('f'), 'a': new_accuracy}
            comb_pred_accuracy_dict.append(d)

        else:
            combined_predict.append(0)

            d = {'t': 'e', 'f': unet_dict.get('f'), 'a': new_accuracy}
            comb_pred_accuracy_dict.append(d)
        
    combined_predict = np.array(combined_predict)

    # Compute combined accuracy
    correct = 0
    total = 0

    with torch.no_grad():
        total = test_classes.size(0)
        correct = (combined_predict == test_classes.numpy()).sum().item()

    acc = (correct / total)

    acu = "%.4f" % acc
    print('Accuracy on '+ str(total) +' testset: ' + str(acu))

    return acu
