import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import os
import glob
import json
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer

BUILD_LSPC = True

def process_df_columns_to_wordocc(file, columns_preprocess_wordcooc, feature_combinations):
    data_df = None
    if '.pkl.gz' in file:
        data_df = pd.read_pickle(file)
    elif '.json.gz' in file:
        data_df = pd.read_json(file, lines=True)
    else:
        print(f'unrecognized file format: {Path(file).suffix}')
    data_df.fillna('', inplace=True)

    # preprocess selected columns
    for column in columns_preprocess_wordcooc:
        data_df[column] = data_df[column].astype(str)

    # build combined features for every feature combination
    for feature_combination in feature_combinations:
        feats_to_combine = feature_combination.split('+')
        data_df[feature_combination + '_wordocc_left'] = data_df[feats_to_combine[0] + '_left']
        data_df[feature_combination + '_wordocc_right'] = data_df[feats_to_combine[0] + '_right']

        for feat_to_combine in feats_to_combine[1:]:
            data_df[feature_combination + '_wordocc_left'] += (' ' + data_df[feat_to_combine + '_left'])
            data_df[feature_combination + '_wordocc_right'] += (' ' + data_df[feat_to_combine + '_right'])

        data_df[feature_combination + '_wordocc_left'] = data_df[feature_combination + '_wordocc_left'].str.strip()
        data_df[feature_combination + '_wordocc_right'] = data_df[feature_combination + '_wordocc_right'].str.strip()

    return data_df


def transform_columns_to_wordcount(data_df, feature_combinations, test_df):
    words = {}

    for feature_combination in feature_combinations:

        # build relevant strings for vocabulary
        all_left_strings = data_df[['id_left', feature_combination + '_wordocc_left']].copy()
        all_left_strings = all_left_strings.rename(
            columns={'id_left': 'id', feature_combination + '_wordocc_left': feature_combination})
        all_right_strings = data_df[['id_right', feature_combination + '_wordocc_right']].copy()
        all_right_strings = all_right_strings.rename(
            columns={'id_right': 'id', feature_combination + '_wordocc_right': feature_combination})
        all_unique_strings = pd.concat([all_left_strings, all_right_strings])
        all_unique_strings = all_unique_strings.drop_duplicates(subset='id')

        # learn vocabulary
        count_vectorizer = CountVectorizer(min_df=2, binary=True)
        count_vectorizer.fit(all_unique_strings[feature_combination])

        words[feature_combination] = count_vectorizer.get_feature_names()

        # apply binary word occurrence
        left_matrix = count_vectorizer.transform(data_df[feature_combination + '_wordocc_left'])
        right_matrix = count_vectorizer.transform(data_df[feature_combination + '_wordocc_right'])
        data_df[feature_combination + '_wordocc_left'] = [x for x in left_matrix]
        data_df[feature_combination + '_wordocc_right'] = [x for x in right_matrix]

        if not isinstance(test_df, type(None)):
            left_matrix_test = count_vectorizer.transform(test_df[feature_combination + '_wordocc_left'])
            right_matrix_test = count_vectorizer.transform(test_df[feature_combination + '_wordocc_right'])
            test_df[feature_combination + '_wordocc_left'] = [x for x in left_matrix_test]
            test_df[feature_combination + '_wordocc_right'] = [x for x in right_matrix_test]

    return data_df, test_df, words


def transform_columns_to_wordcooc(data_df, feature_combinations, test_df):
    for feature_combination in feature_combinations:
        data_df[feature_combination + '_wordcooc'] = list(
            map(lambda x, y: x.multiply(y).astype(int), data_df[feature_combination + '_wordocc_left'].values,
                data_df[feature_combination + '_wordocc_right'].values))

        if not isinstance(test_df, type(None)):
            test_df[feature_combination + '_wordcooc'] = list(
                map(lambda x, y: x.multiply(y).astype(int), test_df[feature_combination + '_wordocc_left'].values,
                    test_df[feature_combination + '_wordocc_right'].values))

    return data_df, test_df


def preprocess_wordcooc(file, columns_to_preprocess, feature_combinations, experiment_name, valid_set=None,
                        test_set=None):
    columns_preprocess_wordcooc = [col + '_left' for col in columns_to_preprocess]
    columns_preprocess_wordcooc.extend([col + '_right' for col in columns_to_preprocess])

    main_df = process_df_columns_to_wordocc(file, columns_preprocess_wordcooc, feature_combinations)

    if not isinstance(test_set, type(None)):
        test_df = process_df_columns_to_wordocc(test_set, columns_preprocess_wordcooc, feature_combinations)
    else:
        test_df = None

    main_df, test_df, words = transform_columns_to_wordcount(main_df, feature_combinations, test_df)
    main_df, test_df = transform_columns_to_wordcooc(main_df, feature_combinations, test_df)

    main_name = os.path.basename(file)
    new_main_name = main_name.replace('.pkl.gz', '_wordcooc')

    out_path = '../../../data/processed/wdc-lspc/wordcooc/{}/'.format(experiment_name)

    os.makedirs(out_path + 'feature-names/', exist_ok=True)

    with open(out_path + 'feature-names/' + new_main_name + '_words.json', 'w') as f:
        json.dump(words, f, ensure_ascii=False)

    if isinstance(valid_set, type(None)):
        main_df.to_pickle(out_path + new_main_name + '.pkl.gz', compression='gzip')
    else:
        validation_ids_df = pd.read_csv(valid_set)
        validation_df = main_df[main_df['pair_id'].isin(validation_ids_df['pair_id'].values)]

        main_df.to_pickle(out_path + new_main_name + '.pkl.gz', compression='gzip')
        valid_name = new_main_name.replace('train', 'valid')
        validation_df.to_pickle(out_path + valid_name + '.pkl.gz', compression='gzip')

    if not isinstance(test_df, type(None)):
        test_name = os.path.basename(test_set)
        test_name = test_name.replace('.pkl.gz', '')
        new_test_name = new_main_name + '_' + test_name

        test_df.to_pickle(out_path + new_test_name + '.pkl.gz', compression='gzip')


columns_to_preprocess = ['title', 'description', 'brand', 'specTableContent']
feature_combinations = ['title', 'title+description', 'title+description+brand',
                        'title+description+brand+specTableContent']

if __name__ == '__main__':

    if BUILD_LSPC:
        for file in glob.glob('../../../data/interim/wdc-lspc/training-sets/*'):

            valid = file.replace('training', 'validation')
            valid = valid.replace('train', 'valid')
            valid = valid.replace('.pkl.gz', '.csv')
            valid = valid.replace('preprocessed_', '')
            valid = valid.replace('interim', 'raw')

            test_cat = os.path.basename(file).split('_')[1]
            test ='../../../data/interim/wdc-lspc/gold-standards/preprocessed_{}_gs.pkl.gz'.format(test_cat)

            preprocess_wordcooc(file, columns_to_preprocess, feature_combinations, experiment_name='learning-curve', valid_set=valid, test_set=test)
            