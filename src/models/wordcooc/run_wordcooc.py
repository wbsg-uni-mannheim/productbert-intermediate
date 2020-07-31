import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import scipy

import os
import time
import glob
import json

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit

from sklearn.metrics import classification_report

import xgboost as xgb

classifiers = {'NaiveBayes':  {'clf':BernoulliNB(),
                            'params':{}},
                   'XGBoost': {'clf':xgb.XGBClassifier(random_state=42, n_jobs=4),
                                'params':{"learning_rate": [0.1, 0.01, 0.001],
                           "gamma" : [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
                           "max_depth": [2, 4, 7, 10],
                           "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
                           "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
                           "reg_alpha": [0, 0.5, 1],
                           "reg_lambda": [1, 1.5, 2, 3, 4.5],
                           "min_child_weight": [1, 3, 5, 7],
                           "n_estimators": [100]}},
                   'RandomForest':  {'clf':RandomForestClassifier(random_state=42, n_jobs=4),
                                'params':{'n_estimators': [100],
                                 'max_features': ['sqrt', 'log2', None],
                                 'max_depth': [2,4,7,10],
                                 'min_samples_split': [2, 5, 10, 20],
                                 'min_samples_leaf': [1, 2, 4, 8],
                                 'class_weight':[None, 'balanced_subsample']
                                 }},
                   'DecisionTree':  {'clf':DecisionTreeClassifier(random_state=42),
                                'params':{'max_features': ['sqrt', 'log2', None],
                                 'max_depth': [2,4,7,10],
                                 'min_samples_split': [2, 5, 10, 20],
                                 'min_samples_leaf': [1, 2, 4, 8],
                                 'class_weight':[None, 'balanced']
                                 }},
                   'LinearSVC':  {'clf':LinearSVC(random_state=42, dual=False),
                      'params':{'C': [0.0001 ,0.001, 0.01, 0.1, 1, 10, 100, 1000],
                      'class_weight':[None, 'balanced']}},
                   'LogisticRegression': {'clf':LogisticRegression(random_state=42, solver='liblinear'),
                        'params':{'C': [0.0001 ,0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'class_weight':[None, 'balanced']}},
                   }

def run_wordcooc(train_set, valid_set, test_set, feature_combinations, classifiers, experiment_name,
                 write_test_set_for_inspection=False):
    train_path = os.path.dirname(train_set)
    train_file = os.path.basename(train_set)
    test_path = os.path.dirname(test_set)
    test_file = os.path.basename(test_set)
    report_train_name = train_file.replace('.pkl.gz', '')
    report_test_name = test_file.replace('.pkl.gz', '')

    os.makedirs(os.path.dirname('../../../reports/wordcooc/{}/'.format(experiment_name)),
                exist_ok=True)

    try:
        os.remove('../../../reports/wordcooc/{}/{}_{}.csv'.format(experiment_name, report_train_name, report_test_name))
    except OSError:
        pass

    with open('../../../reports/wordcooc/{}/{}_{}.csv'.format(experiment_name, report_train_name, report_test_name),
              "w") as f:
        f.write(
            'feature#####model#####mean_train_score#####std_train_score#####mean_valid_score#####std_valid_score#####precision_test#####recall_test#####f1_test#####best_params#####train_time#####prediction_time#####feature_importance#####experiment_name#####train_set#####test_set\n')
    for run in range(1, 4):
        for feature_combination in feature_combinations:

            train_original_df = pd.read_pickle(train_set, compression='gzip')
            gs_df = pd.read_pickle(test_set, compression='gzip')

            feature_file_name = train_file.replace('.pkl.gz', '_words.json')

            with open(train_path + '/feature-names/' + feature_file_name) as json_data:
                words = json.load(json_data)

            validation_ids_df = pd.read_pickle(valid_set, compression='gzip')
            val_df = train_original_df[train_original_df['pair_id'].isin(validation_ids_df['pair_id'].values)]
            train_only_df = train_original_df[~train_original_df['pair_id'].isin(validation_ids_df['pair_id'].values)]
            train_only_df = train_only_df.sample(frac=1, random_state=42)

            pos_neg = train_original_df['label'].value_counts()
            pos_neg = round(pos_neg[0] / pos_neg[1])

            train_ind = []
            val_ind = []

            for i in range(len(train_only_df) - 1):
                train_ind.append(-1)

            for i in range(len(val_df) - 1):
                val_ind.append(0)

            ps = PredefinedSplit(test_fold=np.concatenate((train_ind, val_ind)))

            train_df = pd.concat([train_only_df, val_df])

            for k, v in classifiers.items():

                classifier = v['clf']
                if 'random_state' in classifier.get_params().keys():
                    classifier = classifier.set_params(**{'random_state': run})

                # add pos_neg ratio to XGBoost params
                if k == 'XGBoost':
                    v['params']['scale_pos_weight']: [1, pos_neg]

                model = RandomizedSearchCV(cv=ps, estimator=classifier, param_distributions=v['params'],
                                           random_state=42, n_jobs=4, scoring='f1', n_iter=500, pre_dispatch=8,
                                           return_train_score=True)

                feats_train = scipy.sparse.vstack(train_df[feature_combination + '_wordcooc'])
                labels_train = train_df['label']
                feats_gs = scipy.sparse.vstack(gs_df[feature_combination + '_wordcooc'])
                labels_gs = gs_df['label']

                model.fit(feats_train, labels_train)

                parameters = model.best_params_

                score_names = ['mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score']
                scores = {}
                score_string = ''
                for name in score_names:
                    scores[name] = model.cv_results_[name][model.best_index_]
                    score_string = score_string + name + ': ' + str(scores[name]) + ' '

                if k == 'LogisticRegression' or k == 'LinearSVC':
                    most_important_features = model.best_estimator_.coef_
                    word_importance = zip(words[feature_combination], most_important_features[0].tolist())
                    word_importance = sorted(word_importance, key=lambda importance: importance[1], reverse=True)
                if k == 'RandomForest' or k == 'DecisionTree':
                    most_important_features = model.best_estimator_.feature_importances_
                    word_importance = zip(words[feature_combination], most_important_features.tolist())
                    word_importance = sorted(word_importance, key=lambda importance: importance[1], reverse=True)
                if k == 'NaiveBayes':
                    word_importance = ''
                if k == 'XGBoost':
                    most_important_features = model.best_estimator_.feature_importances_
                    word_importance = zip(words[feature_combination], most_important_features.tolist())
                    word_importance = sorted(word_importance, key=lambda importance: importance[1], reverse=True)

                if k == 'LogisticRegression':
                    learner = LogisticRegression(random_state=run, solver='liblinear', **parameters)
                elif k == 'NaiveBayes':
                    learner = BernoulliNB()
                elif k == 'DecisionTree':
                    learner = DecisionTreeClassifier(random_state=run, **parameters)
                elif k == 'LinearSVC':
                    learner = LinearSVC(random_state=run, dual=False, **parameters)
                elif k == 'RandomForest':
                    learner = RandomForestClassifier(random_state=run, n_jobs=4, **parameters)
                elif k == 'XGBoost':
                    learner = xgb.XGBClassifier(random_state=run, n_jobs=4, **parameters)
                else:
                    print('Learner is not a valid option')
                    break

                model = learner
                feats_train = scipy.sparse.vstack(train_only_df[feature_combination + '_wordcooc'])
                labels_train = train_only_df['label']

                start = time.time()
                model.fit(feats_train, labels_train)
                end = time.time()

                train_time = end - start

                start = time.time()
                preds_gs = model.predict(feats_gs)

                end = time.time()

                pred_time = end - start

                gs_report = classification_report(labels_gs, preds_gs, output_dict=True)

                if write_test_set_for_inspection:

                    out_path = '../../../data/processed/wdc-lspc/inspection/{}/wordcooc/'.format(experiment_name)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    file_name = '_'.join(
                        [os.path.basename(train_set), os.path.basename(test_set), k, feature_combination])
                    file_name = file_name.replace('.csv', '')
                    file_name += f'_{run}.pkl.gz'

                    test_inspection_df = gs_df.copy()
                    if k == 'LinearSVC':
                        proba_gs = model.decision_function(feats_gs).tolist()
                    else:
                        proba_gs = model.predict_proba(feats_gs).tolist()
                    test_inspection_df['pred'] = preds_gs
                    test_inspection_df['Class Prob'] = proba_gs
                    test_inspection_df.to_pickle(out_path + file_name, compression='gzip')

                with open('../../../reports/wordcooc/{}/{}_{}.csv'.format(experiment_name, report_train_name,
                                                                          report_test_name), "a") as f:
                    f.write(feature_combination + '#####' + k + '#####' + str(
                        scores['mean_train_score']) + '#####' + str(scores['std_train_score'])
                            + '#####' + str(scores['mean_test_score']) + '#####' + str(
                        scores['std_test_score']) + '#####' + str(gs_report['1']['precision']) + '#####' + str(
                        gs_report['1']['recall']) + '#####' + str(gs_report['1']['f1-score'])
                            + '#####' + str(parameters) + '#####' + str(train_time) + '#####' + str(pred_time)
                            + '#####' + str(word_importance[
                                            0:100]) + '#####' + experiment_name + '#####' + report_train_name + '#####' + report_test_name + '\n')

if __name__ == '__main__':

    # learning-curve experiment
    feature_combinations = ['title', 'title+description', 'title+description+brand',
                            'title+description+brand+specTableContent']
    experiment_name = 'learning-curve'

    for file in glob.glob('../../../data/processed/wdc-lspc/wordcooc/learning-curve/*'):
        if 'train_' in file and '_gs' not in file:
            valid = file.replace('train_', 'valid_')

            test_cat = '_'.join(os.path.basename(file).split('_')[:2])
            test = os.path.basename(file)
            test = test.replace('.pkl.gz', '_{}_gs.pkl.gz'.format(test_cat))
            test = '../../../data/processed/wdc-lspc/wordcooc/learning-curve/{}'.format(test)

            run_wordcooc(file, valid, test, feature_combinations, classifiers, experiment_name,
                         write_test_set_for_inspection=True)
