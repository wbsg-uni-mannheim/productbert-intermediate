import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import os
import sys
import time
import glob

import deepmatcher as dm

from src.models.deepmatcher import run_deepmatcher

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sys import argv

gpu_id = argv[1]

for file in glob.glob('../../../data/processed/wdc-lspc/magellan/learning-curve/formatted/*'):
    if 'computers_trainonly_xlarge' in file and 'metadata' not in file and '_pairs_' in file:
            
        train_set = file
        valid_set = file.replace('trainonly','valid')
        test_set = '../../../data/processed/wdc-lspc/magellan/learning-curve/formatted/preprocessed_computers_gs_magellan_pairs_formatted.csv'
        pred_set = [
            '../../../data/processed/wdc-lspc/magellan/learning-curve/formatted/preprocessed_computers_gs_magellan_pairs_formatted.csv']

        experiment_name = 'wdc-lspc'
        epochs = 50
        pos_neg_ratio = 6
        batch_size = 16
        lr = 0.001
        lr_decay = 0.8
        embedding = 'fasttext.en.bin'
        nn_type = 'rnn'
        comp_type = 'abs-diff'
        special_name = 'standard'
        features = ['title', 'description', 'brand', 'specTableContent']

        run_deepmatcher.run_dm_model(train_set, valid_set, test_set, experiment_name, gpu_id, epochs, pos_neg_ratio,
                                     batch_size, lr, lr_decay, embedding, nn_type, comp_type, special_name, features,
                                     1, prediction_sets=pred_set)
        run_deepmatcher.run_dm_model(train_set, valid_set, test_set, experiment_name, gpu_id, epochs, pos_neg_ratio,
                                     batch_size, lr, lr_decay, embedding, nn_type, comp_type, special_name, features,
                                     2, prediction_sets=pred_set)
        run_deepmatcher.run_dm_model(train_set, valid_set, test_set, experiment_name, gpu_id, epochs, pos_neg_ratio,
                                     batch_size, lr, lr_decay, embedding, nn_type, comp_type, special_name, features,
                                     3, prediction_sets=pred_set)
