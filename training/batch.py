from training.data_loader import *
import numpy as np
import math
import torch
import torchvision


BATCH_SIZE = 11
class_arr = [0,1]
data_source = 'ounet' # ounet: unet output data, native: 


def generate_training_batch():
    # get generated feature map and classes
    feature_map, classes = load_training_dataset()

    # Generate batch of features
    group = len(feature_map)/BATCH_SIZE

    batch_features = np.split(feature_map, group)
    batch_features = np.array(batch_features)
    batch_features = torch.from_numpy(batch_features).float()

    print(batch_features.shape)

    # Generate batch of classes
    group = len(classes)/BATCH_SIZE

    batch_classes = np.split(classes, group)
    batch_classes = np.array(batch_classes)
    batch_classes = torch.from_numpy(batch_classes).long()

    print(batch_classes.shape)

    return batch_features, batch_classes, classes
