import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import os
import glob
import py_entitymatching as em
from pathlib import Path

BUILD_LSPC = True

def preprocess_magellan(file, columns_to_preprocess, experiment_name, validation_set=None):
    columns_preprocess_magellan = ['ltable_' + col for col in columns_to_preprocess]
    columns_preprocess_magellan.extend(['rtable_' + col for col in columns_to_preprocess])
    data_df = None
    if '.pkl.gz' in file:
        data_df = pd.read_pickle(file)
    elif '.json.gz' in file:
        data_df = pd.read_json(file, lines=True)
    else:
        print(f'unrecognized file format: {Path(file).suffix}')
    data_df.fillna('', inplace=True)
    if 'price' in columns_to_preprocess:
        data_df['price_left'] = data_df['price_left'].replace(r'^\s*$', np.nan, regex=True)
        data_df['price_right'] = data_df['price_right'].replace(r'^\s*$', np.nan, regex=True)
        data_df['price_left'] = data_df['price_left'].astype('float64')
        data_df['price_right'] = data_df['price_right'].astype('float64')
    # change column naming to magellan format
    cols = list(data_df.columns)
    for i, col in enumerate(cols):
        if '_left' in col:
            col = col.replace('_left', '')
            cols[i] = 'ltable_' + col
        if '_right' in col:
            col = col.replace('_right', '')
            cols[i] = 'rtable_' + col
    data_df.columns = cols

    # build left and right subsets
    left_df = data_df[[col for col in data_df.columns if 'ltable_' in col]].copy()
    left_df.drop_duplicates(subset='ltable_id', inplace=True)
    right_df = data_df[[col for col in data_df.columns if 'rtable_' in col]].copy()
    right_df.drop_duplicates(subset='rtable_id', inplace=True)

    # assign magellan ids in subsets
    left_df['mag_id'] = range(0, len(left_df))
    right_df['mag_id'] = range(0, len(right_df))

    # use magellan ids and assign global pair id
    len_assert = len(data_df)
    data_df = data_df.merge(left_df[['ltable_id', 'mag_id']], how='left', on='ltable_id')
    data_df.rename(columns={'mag_id': 'ltable_mag_id'}, inplace=True)
    data_df = data_df.merge(right_df[['rtable_id', 'mag_id']], how='left', on='rtable_id')
    data_df.rename(columns={'mag_id': 'rtable_mag_id'}, inplace=True)
    data_df['_id'] = range(0, len(data_df))
    assert len(data_df) == len_assert

    left_df.drop(columns='ltable_id', inplace=True)
    right_df.drop(columns='rtable_id', inplace=True)

    left_cols = left_df.columns
    left_df.columns = [col.replace('ltable_', '') for col in left_cols]

    right_cols = right_df.columns
    right_df.columns = [col.replace('rtable_', '') for col in right_cols]

    file_name = os.path.basename(file)
    new_file_name = file_name.replace('.pkl.gz', '_magellan_')

    out_path1 = '../../../data/processed/wdc-lspc/magellan/{}/'.format(experiment_name)
    out_path2 = '../../../data/processed/wdc-lspc/magellan/{}/formatted/'.format(experiment_name)

    os.makedirs(out_path2, exist_ok=True)

    left_df.to_csv(out_path1 + new_file_name + 'left.csv.gz', compression='gzip', header=True, index=False)
    right_df.to_csv(out_path1 + new_file_name + 'right.csv.gz', compression='gzip', header=True, index=False)
    data_df.to_csv(out_path1 + new_file_name + 'pairs.csv.gz', compression='gzip', header=True, index=False)

    # magellan formatting for py_entitymatching
    A = em.read_csv_metadata(out_path1 + new_file_name + 'left.csv.gz', key='mag_id')
    em.to_csv_metadata(A, out_path2 + new_file_name + 'left_formatted.csv')
    B = em.read_csv_metadata(out_path1 + new_file_name + 'right.csv.gz', key='mag_id')
    em.to_csv_metadata(B, out_path2 + new_file_name + 'right_formatted.csv')

    C = em.read_csv_metadata(out_path1 + new_file_name + 'pairs.csv.gz',
                             key='_id',
                             ltable=A, rtable=B,
                             fk_ltable='ltable_mag_id', fk_rtable='rtable_mag_id')

    if isinstance(validation_set, type(None)):

        em.to_csv_metadata(C, out_path2 + new_file_name + 'pairs_formatted.csv')

    else:
        validation_ids_df = pd.read_csv(validation_set)
        validation_df = C[C['pair_id'].isin(validation_ids_df['pair_id'].values)]
        train_df = C[~C['pair_id'].isin(validation_ids_df['pair_id'].values)]

        em.to_csv_metadata(C, out_path2 + new_file_name + 'pairs_formatted.csv')

        new_file_name = new_file_name.replace('train', 'trainonly')

        em.to_csv_metadata(train_df, out_path2 + new_file_name + 'pairs_formatted.csv')

        valid_name = new_file_name.replace('trainonly', 'valid')

        em.to_csv_metadata(validation_df, out_path2 + valid_name + 'pairs_formatted.csv')

if __name__ == '__main__':
    if BUILD_LSPC:
        columns_to_preprocess = ['title', 'description', 'brand', 'specTableContent']
        # learning-curve experiment
        for file in glob.glob('../../../data/interim/wdc-lspc/training-sets/*'):
            valid = file.replace('training', 'validation')
            valid = valid.replace('train', 'valid')
            valid = valid.replace('.pkl.gz', '.csv')
            valid = valid.replace('interim', 'raw')
            valid = valid.replace('preprocessed_', '')
            preprocess_magellan(file, columns_to_preprocess, experiment_name='learning-curve', validation_set=valid)

        for file in glob.glob('../../../data/interim/wdc-lspc/gold-standards/*'):
            preprocess_magellan(file, columns_to_preprocess, experiment_name='learning-curve')
