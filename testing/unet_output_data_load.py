# Importing libraries
import numpy as np
import pandas as pd
import math 
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import random

from testing.data_loader import *

data_source = 'ounet'


def prepare_unet_sample_list():
    # patient selection is manual
    # change unet_patients manually here with different values
    unet_patients = 55 #5,7,8,10,12,15,20,30

    test_items = []
    unet_pred_full = []

    unet_accuracies=pd.read_csv('D:/SCA/256_256/roi/confidences.txt', header=None, delimiter="\t")
    unet_accuracies = unet_accuracies.values

    for ua in unet_accuracies:
        sp_res = ua[0].split(',')
        arr = sp_res[0].split('_')
        d = {'t':arr[3], 'f': sp_res[0], 'a': sp_res[1]}
        unet_pred_full.append(d)

    if data_source == 'ounet':
        up_index = []

        for ds in unet_accuracies:
            x = ds[0].split(',')
            cl = x[0].split('_')
            up_index.append(int(cl[0]))

        patients_index = list(set(patients_index)) # total 55 patients
        up_index.sort()
        
        sampled_list = random.sample(up_index, unet_patients)

        for i in range(0, int(len(sampled_list))):
            test_items.append(sampled_list[i])
            
        print(sampled_list)

    return sampled_list, unet_pred_full, test_items


def map_unet_output():
    # Get unet accuracies
    unet_prediction_accuracies = []
    test_feature_map, test_classes, test_files_title = load_test_dataset()
    sampled_list, unet_pred_full, test_items = prepare_unet_sample_list()

    print('Reading...')
    if data_source == 'ounet':
        for item in test_files_title:
            fstr = item.split('.')[0]
            item = next(item for item in unet_pred_full if item["f"] == fstr)
            unet_prediction_accuracies.append(item)
            #print(item)
            
    unet_prediction_accuracies = np.array(unet_prediction_accuracies)
            
    print('Done!')
    return unet_prediction_accuracies


def get_mapped_image_slices():
    test_loc = 'D:/SCA/256_256/feature_map/test_unet'
    test_files = [f for f in listdir(test_loc) if isfile(join(test_loc, f))]
    img_slices = []

    for file in test_files:
        f_arr = file.split('_')
        pno = int(f_arr[0])
        slc = int(f_arr[1])
        img_slices.append(str(pno)+'_'+str(slc))
        
    print(len(set(img_slices)))
    return img_slices
