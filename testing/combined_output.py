# Importing libraries
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os
from PIL import Image


def compute_pfcnn_combined_accuracy(pred_accuracy_dict, unet_prediction_accuracies, t_out, test_classes):
    combined_predict = []
    comb_pred_accuracy_dict = []

    for i in range(0, len(t_out)):
        unet_dict = unet_prediction_accuracies[i]
        native_dict = pred_accuracy_dict[i]
        
        if unet_dict.get('t') == native_dict.get('t'):
            new_accuracy = (float(unet_dict.get('a'))+float(native_dict.get('a')))/2
            
            if native_dict.get('t') == 'c' and new_accuracy > 0.55:
                combined_predict.append(1)
                
                d = {'t': 'c', 'f': unet_dict.get('f'), 'a': new_accuracy}
                comb_pred_accuracy_dict.append(d)
                
            elif native_dict.get('t') == 'c' and new_accuracy < 0.55:
                combined_predict.append(0)
                
                d = {'t': 'e', 'f': unet_dict.get('f'), 'a': new_accuracy}
                comb_pred_accuracy_dict.append(d)
                
            elif native_dict.get('t') == 'e' and new_accuracy < 0.55:
                combined_predict.append(0)
                
                d = {'t': 'e', 'f': unet_dict.get('f'), 'a': new_accuracy}
                comb_pred_accuracy_dict.append(d)
                
            elif native_dict.get('t') == 'e' and new_accuracy > 0.55:
                combined_predict.append(1)
                
                d = {'t': 'c', 'f': unet_dict.get('f'), 'a': new_accuracy}
                comb_pred_accuracy_dict.append(d)
                
                
        else:
            unet_cat_prob = float(unet_dict.get('a'))
            unet_echo_prob = 1-unet_cat_prob

            native_echo_prob = float(native_dict.get('a'))
            native_cat_prob = 1-native_echo_prob

            new_accuracy_cat = (unet_cat_prob+native_cat_prob)/2
            new_accuracy_echo = (unet_echo_prob+native_echo_prob)/2
            
            if new_accuracy_cat > new_accuracy_echo:
                combined_predict.append(1)
                
                d = {'t': 'c', 'f': unet_dict.get('f'), 'a': new_accuracy_cat}
                comb_pred_accuracy_dict.append(d)
                
            else:
                combined_predict.append(0)
                
                d = {'t': 'e', 'f': unet_dict.get('f'), 'a': new_accuracy_echo}
                comb_pred_accuracy_dict.append(d)
                
            # Or try the following
            '''if new_accuracy_cat > 0.55:
                combined_predict.append(1)
            else:
                combined_predict.append(0)'''
        
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
