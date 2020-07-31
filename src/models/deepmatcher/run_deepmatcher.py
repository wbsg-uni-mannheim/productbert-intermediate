import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import os
import sys
import time

import deepmatcher as dm

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def get_features(data_path):
    data_df = pd.read_csv(data_path)
    column_names = data_df.columns.tolist()
    return column_names

def run_dm_model(train_set, valid_set, test_set, experiment_name, gpu_id, epochs, pos_neg_ratio, batch_size, lr, lr_decay, embedding, nn_type, comp_type, special_name, features, run_no, smoothing=0.05, prediction_sets=None):
    
    os.makedirs(os.path.dirname('../../../reports/deepmatcher/raw/{}/'.format(experiment_name)), exist_ok=True)
    os.makedirs(os.path.dirname('../../../cache/deepmatcher/{}/data-cache/'.format(experiment_name)), exist_ok=True)
    os.makedirs(os.path.dirname('../../../cache/deepmatcher/{}/models/'.format(experiment_name)), exist_ok=True)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
    
    dm.data.reset_vector_cache()
    
    ignore_columns = get_features(train_set)
    left_right_features = ['ltable_'+ feat for feat in features]
    left_right_features.extend(['rtable_'+ feat for feat in features])
    for feat in left_right_features:
        ignore_columns.remove(feat)
        
    features_filename = '-'.join(features)
    train_set_filename = os.path.basename(train_set)
    train_set_filename = train_set_filename.replace('.csv','')
    train, valid, test = dm.data.process(
        path='',
        cache='../../../cache/deepmatcher/{}/data-cache/{}.pth'.format(experiment_name, train_set_filename+'_'+embedding),
        train=train_set,
        validation=valid_set,
        test=test_set,
        embeddings=embedding,
        use_magellan_convention=True,
        ignore_columns=ignore_columns)

    old_stdout = sys.stdout
    
    sys.stdout = open('../../../reports/deepmatcher/raw/{}/{}_{}_{}_epochs{}_ratio{}_batch{}_lr{}_lrdecay{}_{}_{}_{}_run{}.txt'.format(experiment_name,nn_type,comp_type,special_name,epochs,pos_neg_ratio,batch_size,lr,lr_decay,embedding,features_filename,train_set_filename,run_no), 'w')
    model = dm.MatchingModel(attr_summarizer=nn_type, attr_comparator=comp_type)
    model.initialize(train)
    
    optim = dm.optim.Optimizer(method='adam', lr=lr, max_grad_norm=5, start_decay_at=1, beta1=0.9, beta2=0.999, adagrad_accum=0.0, lr_decay=lr_decay)
    optim.set_parameters(model.named_parameters())
    
    start = time.time()
    model.run_train(
        train,
        valid,
        epochs=epochs,
        batch_size=batch_size,
        best_save_path='../../../cache/deepmatcher/{}/models/{}_{}_{}_epochs{}_ratio{}_batch{}_lr{}_lrdecay{}_{}_{}_{}_run{}_model.pth'.format(experiment_name,nn_type,comp_type,special_name,epochs,pos_neg_ratio,batch_size,lr,lr_decay,embedding,features_filename,train_set_filename,run_no),
        pos_neg_ratio=pos_neg_ratio,
        optimizer=optim,
        label_smoothing=smoothing
    )
    end = time.time()
    print('Training time: '+str(end-start))
    start = time.time()
    model.run_eval(test, batch_size=batch_size)
    end = time.time()
    print('Prediction time: '+str(end-start))
    sys.stdout = old_stdout

    if prediction_sets is not None:
        out_path = '../../../data/processed/inspection/{}/deepmatcher/'.format(experiment_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        for prediction_set in prediction_sets:
            candidate = dm.data.process_unlabeled(
            path=prediction_set,
            trained_model=model,
            ignore_columns=ignore_columns)
            predictions = model.run_prediction(candidate, output_attributes=True, batch_size=8)
    
            predictions['label_pred'] = predictions['match_score'].apply(lambda score: 1 if score >= 0.5 else 0)

            file_name = os.path.basename('{}_{}_{}_epochs{}_ratio{}_batch{}_lr{}_lrdecay{}_{}_{}_{}_run{}_model.pth'.format(nn_type,comp_type,special_name,epochs,pos_neg_ratio,batch_size,lr,lr_decay,embedding,features_filename,train_set_filename,run_no))+os.path.basename(prediction_set)
            file_name = file_name.replace('.csv', '.csv.gz')
            file_name = file_name.replace('model.pth', '')
            file_name = file_name.replace('_formatted', '')

            predictions.to_csv(out_path+file_name, compression='gzip', header=True, index=False)

def predict_and_write_for_inspection(test_path, model_path, experiment_name, gpu_id, nn_type, comp_type, features):
    
    out_path = '../../../data/processed/inspection/{}/deepmatcher/'.format(experiment_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    ignore_columns = get_features(test_path)
    left_right_features = ['ltable_'+ feat for feat in features]
    left_right_features.extend(['rtable_'+ feat for feat in features])
    for feat in left_right_features:
        ignore_columns.remove(feat)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
    
    model = dm.MatchingModel(attr_summarizer=nn_type, attr_comparator=comp_type)
    model.load_state(model_path)
    
    candidate = dm.data.process_unlabeled(
    path=test_path,
    trained_model=model,
    ignore_columns=ignore_columns)
    
    predictions = model.run_prediction(candidate, output_attributes=True, batch_size=8)
    
    predictions['pred'] = predictions['match_score'].apply(lambda score: 1 if score >= 0.5 else 0)
    
    print(classification_report(predictions['label'], predictions['pred'], digits=4))
    print(confusion_matrix(predictions['label'], predictions['pred']))
    
    file_name = os.path.basename(model_path)+os.path.basename(test_path)
    file_name = file_name.replace('model.pth', '_')
    file_name = file_name.replace('.csv', '.csv.gz')
    file_name = file_name.replace('_formatted', '')
    
    predictions.to_csv(out_path+file_name, compression='gzip', header=True, index=False)
