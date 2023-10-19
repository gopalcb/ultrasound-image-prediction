import numpy as np
import pandas as pd
import random
import os
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torchvision


def select_random_patient_test(select_patients=100):
    # Random data set select
    patients_index = []
    test_items = []

    last_index = 0

    for i in range(1, 101):
        patients_index.append(i)

    sampled_list = random.sample(patients_index, select_patients)

    for i in range(i+1, len(sampled_list)):
        test_items.append(sampled_list[i])

    print(sampled_list)
    return sampled_list, test_items


def load_test_dataset():
    # Load test data
    # Veriables
    test_feature_map = []
    test_classes = [] # Catheter: 1, Echo: 0
    test_files_title = []
    data_count = 0

    # Read test data
    test_loc = 'D:/SCA/256_256/feature_map/test_unet'
    test_files = [f for f in listdir(test_loc) if isfile(join(test_loc, f))]
    sampled_list, test_items = select_random_patient_test()

    print('Loading test data...')

    for file in test_files:
        # split file name to get patient details
        f_arr = file.split('_')
        pn = int(f_arr[0])  # patient number
        otype = f_arr[3]  # catheter type
        
        if pn in test_items:
            
            if otype == 'e': # echo
        
                lbl = 1 if otype == 'c' else 0  # c = catheter

                dataset=pd.read_csv(test_loc + '/' + file, header=None, delimiter='\t')   
                dataset = dataset.values
                if len(dataset) == 0:
                    print('null')

                one_object_features = []
                d_count = 0

                for ds in dataset:
                    # Order: magnitude(0), phase(1), signature(2), sector(3), distance(4)

                    if d_count <= 39:
                        o_arr = ds[0].split(',')
                        o_arr = np.array(o_arr)
                        o_arr = o_arr.astype(np.float)

                        # tune here manually for different feature selection
                        # comment/uncomment to get different combination manually
                        one_object_features.append(o_arr[0])
                        one_object_features.append(o_arr[1])
                        one_object_features.append(o_arr[2])
                        one_object_features.append(o_arr[3])
                        one_object_features.append(o_arr[4])

                        d_count += 1

                data_count = data_count + 1
                print('\r' + str(data_count), end='')

                one_object_features = np.array(one_object_features)

                # Data normalization
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(one_object_features.reshape(-1, 1))

                test_feature_map.append(scaled_data)
                test_classes.append(lbl)
                test_files_title.append(file)
        
    test_feature_map = np.array(test_feature_map)
    test_classes = np.array(test_classes)
    test_files_title = np.array(test_files_title)

    test_feature_map = torch.from_numpy(test_feature_map).float()
    test_classes = torch.from_numpy(test_classes).long()

    print('\nDone!')
    return test_feature_map, test_classes, test_files_title
