import numpy as np
import pandas as pd
import random
import os
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def select_random_patient(select_patients=100):
    # Random data set select
    patients_index = []

    train_items = []
    last_index = 0

    for i in range(1, 101):
        patients_index.append(i)

    sampled_list = random.sample(patients_index, select_patients)

    for i in range(0, int(len(sampled_list)/2)):
        train_items.append(sampled_list[i])
        last_index = i

    print(sampled_list)
    return sampled_list, train_items


def load_training_dataset():
    # Veriables
    feature_map = []
    classes = [] # Catheter: 1, Echo: 0
    data_count = 0
    file_count = 0

    mags = []
    phases = []
    sectors = []
    distances = []
    sigs = []

    cat_data = []
    echo_data = []

    # Read train data
    train_loc = 'D:/SCA/256_256/feature_map/exp/train'
    train_files = [f for f in listdir(train_loc) if isfile(join(train_loc, f))]

    sampled_list, train_items = select_random_patient()

    print('Loading train data...')

    for file in train_files:
        # split file name to get patient details
        f_arr = file.split('_')
        pn = int(f_arr[0])  # patient number
        otype = f_arr[3]  # catheter type
        
        if pn in train_items:

            lbl = 1 if otype == 'c' else 0

            dataset=pd.read_csv(train_loc + '/' + file, header=None, delimiter='\t')   
            dataset = dataset.values

            one_object_features = []

            d_count = 0
            # load and populate features
            for ds in dataset:
                # Order: magnitude(0), phase(1), signature(2), sector(3), distance(4)
                if d_count <= 39:

                    o_arr = ds[0].split(',')
                    o_arr = np.array(o_arr)
                    o_arr = o_arr.astype(np.float)

                    # load magnitude, phase angle, sector index and distance
                    mags.append(o_arr[0])
                    phases.append(o_arr[1])
                    sectors.append(o_arr[3])
                    distances.append(o_arr[4])
                    sigs.append(o_arr[2])

                    # tune here manually for different feature selection
                    # comment/uncomment to get different combination manually
                    one_object_features.append(o_arr[0])
                    one_object_features.append(o_arr[1])
                    one_object_features.append(o_arr[2])
                    one_object_features.append(o_arr[3])
                    one_object_features.append(o_arr[4])

                    d_count += 1

            one_object_features = np.array(one_object_features)

            # Data normalization
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(one_object_features.reshape(-1, 1))

            feature_map.append(scaled_data)
            classes.append(lbl)

            # if label is 1 > then catheter
            # otherwise echo
            if lbl == 1:
                cat_data.append(scaled_data)
                echo_data.append(scaled_data)
            else:
                echo_data.append(scaled_data)

            data_count = data_count + 1
            print('\r' + str(data_count), end='')

            file_count += 1
        
    feature_map = np.array(feature_map)
    classes = np.array(classes)

    print('\nDone!')
    return feature_map, classes