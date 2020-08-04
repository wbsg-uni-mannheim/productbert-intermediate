import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt

from copy import deepcopy

import torch

from transformers import BertTokenizer

from sklearn.preprocessing import LabelEncoder

# process data required for fine-tuning
BUILD_LSPC = True
# process intermediate training sets if you want to replicate intermediate training
BUILD_PRETRAIN_COMPUTERS_AND_4CAT = False
BUILD_PRETRAIN_COMPUTERS_AND_4CAT_MASKED = False


def process_to_bert(dataset, attributes, tokenizer, seq_length, seq_length_titleonly, comb_func, cutting_func=None,
                    mlm=False, multi_encoder=None):
    dataset = dataset.fillna('')

    if multi_encoder is None:
        try:
            cluster_id_set_left = set()
            cluster_id_set_left.update(dataset['cluster_id_left'].tolist())
            cluster_id_set_right = set()
            cluster_id_set_right.update(dataset['cluster_id_right'].tolist())
            cluster_id_set_left.update(cluster_id_set_right)
            dataset = dataset.rename(columns={'cluster_id_left': 'label_multi1', 'cluster_id_right': 'label_multi2'})
            label_enc = LabelEncoder()
            label_enc.fit(list(cluster_id_set_left))
            dataset['label_multi1'] = label_enc.transform(dataset['label_multi1'])
            dataset['label_multi2'] = label_enc.transform(dataset['label_multi2'])

        except KeyError:
            pass
    else:
        dataset = dataset.rename(columns={'cluster_id_left': 'label_multi1', 'cluster_id_right': 'label_multi2'})
        try:
            dataset['label_multi1'] = multi_encoder.transform(dataset['label_multi1'])
            dataset['label_multi2'] = multi_encoder.transform(dataset['label_multi2'])
        except ValueError:
            dataset['label_multi1'] = 0
            dataset['label_multi2'] = 0

    print(f'Before cutting:')
    _print_attribute_stats(dataset, attributes)
    if cutting_func:
        tqdm.pandas(desc='Cutting attributes')
        dataset = dataset.progress_apply(cutting_func, axis=1)
        print(f'After cutting:')
        _print_attribute_stats(dataset, attributes)

    dataset['sequence_left'], dataset['sequence_left_titleonly'], dataset['sequence_right'], dataset[
        'sequence_right_titleonly'] = comb_func(dataset)

    dataset['sequence_left'] = dataset['sequence_left'].str.split()
    dataset['sequence_left'] = dataset['sequence_left'].str.join(' ')
    dataset['sequence_right'] = dataset['sequence_right'].str.split()
    dataset['sequence_right'] = dataset['sequence_right'].str.join(' ')

    tqdm.pandas(desc='Encoding left sequence')
    dataset['sequence_left_encoded'] = dataset['sequence_left'].progress_apply(
        lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)))
    dataset['sequence_left_titleonly_encoded'] = dataset['sequence_left_titleonly'].progress_apply(
        lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)))
    tqdm.pandas(desc='Encoding right sequence')
    dataset['sequence_right_encoded'] = dataset['sequence_right'].progress_apply(
        lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)))
    dataset['sequence_right_titleonly_encoded'] = dataset['sequence_right_titleonly'].progress_apply(
        lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)))

    tqdm.pandas(desc='Tokenizing left sequence for inspection')
    dataset['sequence_left_inspect'] = dataset['sequence_left'].progress_apply(lambda x: tokenizer.tokenize(x))
    dataset['sequence_left_titleonly_inspect'] = dataset['sequence_left_titleonly'].progress_apply(
        lambda x: tokenizer.tokenize(x))
    tqdm.pandas(desc='Tokenizing right sequence for inspection')
    dataset['sequence_right_inspect'] = dataset['sequence_right'].progress_apply(lambda x: tokenizer.tokenize(x))
    dataset['sequence_right_titleonly_inspect'] = dataset['sequence_right_titleonly'].progress_apply(
        lambda x: tokenizer.tokenize(x))

    dataset_combined_length = dataset.apply(
        lambda x: len(x['sequence_left_inspect']) + len(x['sequence_right_inspect']), axis=1)
    dataset_combined_length_binned = pd.cut(dataset_combined_length, [-1, 32, 64, 128, 256, 512, 50000],
                                            labels=['32', '64', '128', '256', '512', '50000'])
    print('Full sequence:')
    plt.hist(dataset_combined_length_binned)
    plt.show()

    dataset_combined_length = dataset.apply(
        lambda x: len(x['sequence_left_titleonly_inspect']) + len(x['sequence_right_titleonly_inspect']), axis=1)
    dataset_combined_length_binned = pd.cut(dataset_combined_length, [-1, 32, 64, 128, 256, 512, 50000],
                                            labels=['32', '64', '128', '256', '512', '50000'])
    print('Title only sequence:')
    plt.hist(dataset_combined_length_binned)
    plt.show()

    tqdm.pandas(desc='Encoding full BERT sequence')
    dataset['sequence_encoded'] = dataset.progress_apply(
        lambda row: tokenizer.prepare_for_model(row['sequence_left_encoded'], row['sequence_right_encoded'],
                                                max_length=seq_length, pad_to_max_length=True), axis=1)
    dataset['sequence_titleonly_encoded'] = dataset.progress_apply(
        lambda row: tokenizer.prepare_for_model(row['sequence_left_titleonly_encoded'],
                                                row['sequence_right_titleonly_encoded'],
                                                max_length=seq_length_titleonly, pad_to_max_length=True), axis=1)

    try:
        dataset_reduced = dataset[['label', 'label_multi1', 'label_multi2', 'pair_id', 'sequence_encoded']]
        dataset_reduced_titleonly = dataset[
            ['label', 'label_multi1', 'label_multi2', 'pair_id', 'sequence_titleonly_encoded']]
    except KeyError:
        dataset_reduced = dataset[['label', 'pair_id', 'sequence_encoded']]
        dataset_reduced_titleonly = dataset[['label', 'pair_id', 'sequence_titleonly_encoded']]
    tqdm.pandas(desc='Converting sequence to Tensors')
    dataset_reduced = dataset_reduced.progress_apply(_extract_columns, axis=1)
    dataset_reduced_titleonly = dataset_reduced_titleonly.progress_apply(_extract_columns_titleonly, axis=1)

    dataset_reduced.drop(columns='sequence_encoded', inplace=True)
    dataset_reduced_titleonly.drop(columns='sequence_titleonly_encoded', inplace=True)

    if mlm:
        print('Masking Tokens')
        masked_seq, masked_labels = mask_tokens(dataset_reduced['input_ids'], tokenizer)
        masked_seq_titleonly, masked_labels_titleonly = mask_tokens(dataset_reduced_titleonly['input_ids'], tokenizer)

        dataset_reduced['input_ids'] = masked_seq
        dataset_reduced['mlm_labels'] = masked_labels

        dataset_reduced_titleonly['input_ids'] = masked_seq_titleonly
        dataset_reduced_titleonly['mlm_labels'] = masked_labels_titleonly

        dataset_reduced = dataset_reduced.progress_apply(_fix_columns, axis=1)
        dataset_reduced_titleonly = dataset_reduced_titleonly.progress_apply(_fix_columns, axis=1)
        print('Finished Masking Tokens')
    dataset_inspect = dataset[
        ['sequence_left', 'sequence_left_inspect', 'sequence_left_titleonly', 'sequence_left_titleonly_inspect',
         'sequence_right', 'sequence_right_inspect', 'sequence_right_titleonly', 'sequence_right_titleonly_inspect',
         'pair_id']]

    return dataset_reduced, dataset_reduced_titleonly, dataset_inspect


def _att_to_seq_lspc(dataset):
    seq_left = dataset['brand_left'] + ' ' + dataset['title_left'] + ' ' + dataset['description_left'] + ' ' + dataset[
        'specTableContent_left']
    seq_left_titleonly = dataset['title_left']
    seq_right = dataset['brand_right'] + ' ' + dataset['title_right'] + ' ' + dataset['description_right'] + ' ' + \
                dataset['specTableContent_right']
    seq_right_titleonly = dataset['title_right']
    return seq_left, seq_left_titleonly, seq_right, seq_right_titleonly


def _att_to_seq_abtbuy(dataset):
    seq_left = dataset['name_left'] + ' ' + dataset['description_left'] + ' ' + dataset['price_left'].astype(str)
    seq_left_titleonly = dataset['name_left']
    seq_right = dataset['name_right'] + ' ' + dataset['description_right'] + ' ' + dataset['price_right'].astype(str)
    seq_right_titleonly = dataset['name_right']
    return seq_left, seq_left_titleonly, seq_right, seq_right_titleonly


def _att_to_seq_amazongoogle(dataset):
    seq_left = dataset['manufacturer_left'] + ' ' + dataset['name_left'] + ' ' + dataset['description_left'] + ' ' + \
               dataset['price_left'].astype(str)
    seq_left_titleonly = dataset['name_left']
    seq_right = dataset['manufacturer_right'] + ' ' + dataset['name_right'] + ' ' + dataset['description_right'] + ' ' + \
                dataset['price_right'].astype(str)
    seq_right_titleonly = dataset['name_right']
    return seq_left, seq_left_titleonly, seq_right, seq_right_titleonly


def _print_attribute_stats(dataset, attributes):
    for attr in attributes:
        attribute = list(dataset[f'{attr}_left'].values)
        attribute.extend(list(dataset[f'{attr}_right'].values))
        attribute_clean = [x for x in attribute if x != '']
        attribute_tokens = [x.split(' ') for x in attribute_clean]
        att_len = [len(x) for x in attribute_tokens]
        att_len_max = max(att_len)
        att_len_avg = np.mean(att_len)
        att_len_median = np.median(att_len)
        print(f'{attr}: Max length: {att_len_max}, mean length: {att_len_avg}, median length: {att_len_median}')


def _cut_lspc(row):
    row[f'title_left'] = ' '.join(row[f'title_left'].split(' ')[:50])
    row[f'title_right'] = ' '.join(row[f'title_right'].split(' ')[:50])
    row[f'brand_left'] = ' '.join(row[f'brand_left'].split(' ')[:5])
    row[f'brand_right'] = ' '.join(row[f'brand_right'].split(' ')[:5])
    row[f'description_left'] = ' '.join(row[f'description_left'].split(' ')[:100])
    row[f'description_right'] = ' '.join(row[f'description_right'].split(' ')[:100])
    row[f'specTableContent_left'] = ' '.join(row[f'specTableContent_left'].split(' ')[:200])
    row[f'specTableContent_right'] = ' '.join(row[f'specTableContent_right'].split(' ')[:200])
    return row


def _cut_amazongoogle(row):
    row[f'description_left'] = ' '.join(row[f'description_left'].split(' ')[:100])
    row[f'description_right'] = ' '.join(row[f'description_right'].split(' ')[:100])
    return row


def _extract_columns(row):
    row['input_ids'] = torch.tensor(row['sequence_encoded']['input_ids'])
    row['token_type_ids'] = torch.tensor(row['sequence_encoded']['token_type_ids'])
    row['attention_mask'] = torch.tensor(row['sequence_encoded']['attention_mask'])
    try:
        row['label_multi1'] = torch.tensor(row['label_multi1'])
        row['label_multi2'] = torch.tensor(row['label_multi2'])
        row['label'] = torch.tensor(row['label']).unsqueeze(0)
    except KeyError:
        row['label'] = torch.tensor(row['label']).unsqueeze(0)
    return row


def _fix_columns(row):
    row['input_ids'] = row['input_ids'].tolist()
    row['mlm_labels'] = row['mlm_labels'].tolist()
    row['input_ids'] = torch.tensor(row['input_ids'])
    row['mlm_labels'] = torch.tensor(row['mlm_labels'])
    return row


def _extract_columns_titleonly(row):
    row['input_ids'] = torch.tensor(row['sequence_titleonly_encoded']['input_ids'])
    row['token_type_ids'] = torch.tensor(row['sequence_titleonly_encoded']['token_type_ids'])
    row['attention_mask'] = torch.tensor(row['sequence_titleonly_encoded']['attention_mask'])
    try:
        row['label_multi1'] = torch.tensor(row['label_multi1'])
        row['label_multi2'] = torch.tensor(row['label_multi2'])
        row['label'] = torch.tensor(row['label']).unsqueeze(0)
    except KeyError:
        row['label'] = torch.tensor(row['label']).unsqueeze(0)
    return row


def mask_tokens(sequences, tokenizer, probability=0.15):
    inputs = torch.stack(sequences.tolist())
    labels = deepcopy(inputs)
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    inputs = list(torch.unbind(inputs))
    labels = list(torch.unbind(labels))
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def get_encoder(name):
    name_dict = {
        'computers': '../../../data/interim/wdc-lspc/gold-standards/preprocessed_computers_gs.pkl.gz',
        'cameras': '../../../data/interim/wdc-lspc/gold-standards/preprocessed_cameras_gs.pkl.gz',
        'watches': '../../../data/interim/wdc-lspc/gold-standards/preprocessed_watches_gs.pkl.gz',
        'shoes': '../../../data/interim/wdc-lspc/gold-standards/preprocessed_shoes_gs.pkl.gz'
    }
    gs = pd.read_pickle(name_dict[name])
    enc = LabelEncoder()
    cluster_id_set_left = set()
    cluster_id_set_left.update(gs['cluster_id_left'].tolist())
    cluster_id_set_right = set()
    cluster_id_set_right.update(gs['cluster_id_right'].tolist())
    cluster_id_set_left.update(cluster_id_set_right)
    enc.fit(list(cluster_id_set_left))
    return enc

if __name__ == '__main__':
    if BUILD_LSPC:

        encoders = {
            'computers': get_encoder('computers'),
            'cameras': get_encoder('cameras'),
            'watches': get_encoder('watches'),
            'shoes': get_encoder('shoes'),
        }

        Path('../../../data/processed/wdc-lspc/bert/inspection/').mkdir(parents=True, exist_ok=True)
        path = '../../../data/processed/wdc-lspc/bert/'

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        attributes = ['title', 'description', 'brand', 'specTableContent']
        seq_length = 512
        seq_length_titleonly = 128

        datasets_lspc_train = ['preprocessed_computers_train_small', 'preprocessed_computers_train_medium',
                               'preprocessed_computers_train_large', 'preprocessed_computers_train_xlarge',
                               'preprocessed_cameras_train_small', 'preprocessed_cameras_train_medium',
                               'preprocessed_cameras_train_large', 'preprocessed_cameras_train_xlarge',
                               'preprocessed_watches_train_small', 'preprocessed_watches_train_medium',
                               'preprocessed_watches_train_large', 'preprocessed_watches_train_xlarge',
                               'preprocessed_shoes_train_small', 'preprocessed_shoes_train_medium',
                               'preprocessed_shoes_train_large', 'preprocessed_shoes_train_xlarge'
                               ]

        datasets_lspc_gs = ['preprocessed_computers_gs', 'preprocessed_cameras_gs',
                            'preprocessed_watches_gs', 'preprocessed_shoes_gs'
                            ]

        for ds in datasets_lspc_gs:
            enc = None
            for key in encoders.keys():
                if key in ds:
                    assert enc is None
                    enc = encoders[key]
            df = pd.read_pickle(f'../../../data/interim/wdc-lspc/gold-standards/{ds}.pkl.gz')
            df_gs, df_gs_titleonly, df_inspect = process_to_bert(df, attributes, tokenizer, seq_length,
                                                                 seq_length_titleonly,
                                                                 _att_to_seq_lspc, _cut_lspc, multi_encoder=enc)

            df_gs.to_pickle(f'{path}{ds}_bert_cutBTDS_{seq_length}.pkl.gz', compression='gzip')
            df_gs_titleonly.to_pickle(f'{path}{ds}_bert_cutT_titleonly_{seq_length_titleonly}.pkl.gz',
                                      compression='gzip')
            df_inspect.to_csv(f'{path}inspection/{ds}_bert_cutBTDS_{seq_length}.csv.gz', index=False)

        for ds in datasets_lspc_train:
            enc = None
            for key in encoders.keys():
                if key in ds:
                    assert enc is None
                    enc = encoders[key]
            df = pd.read_pickle(f'../../../data/interim/wdc-lspc/training-sets/{ds}.pkl.gz')
            df_train, df_train_titleonly, df_inspect = process_to_bert(df, attributes, tokenizer, seq_length,
                                                                       seq_length_titleonly, _att_to_seq_lspc,
                                                                       _cut_lspc, multi_encoder=enc)

            df_train.to_pickle(f'{path}{ds}_bert_cutBTDS_{seq_length}.pkl.gz', compression='gzip')
            df_train_titleonly.to_pickle(f'{path}{ds}_bert_cutT_titleonly_{seq_length_titleonly}.pkl.gz',
                                         compression='gzip')
            df_inspect.to_csv(f'{path}inspection/{ds}_bert_cutBTDS_{seq_length}.csv.gz', index=False)

    ###############################################################

    if BUILD_PRETRAIN_COMPUTERS_AND_4CAT:
        Path('../../../data/processed/wdc-lspc/bert/inspection/').mkdir(parents=True, exist_ok=True)
        path = '../../../data/processed/wdc-lspc/bert/'

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        attributes = ['title', 'description', 'brand', 'specTableContent']
        seq_length = 128
        seq_length_titleonly = 128

        datasets_lspc_pretrain = ['pre-training_computers_only_new_15', 'pre-training_4cat_new_5']

        for ds in datasets_lspc_pretrain:
            df = pd.read_pickle(f'../../../data/raw/wdc-lspc/pre-training-set/{ds}.pkl.gz')
            df_train, df_train_titleonly, df_inspect = process_to_bert(df, attributes, tokenizer, seq_length,
                                                                       seq_length_titleonly, _att_to_seq_lspc, _cut_lspc)

            df_train.to_pickle(f'{path}{ds}_bert_cutBTDS_{seq_length}.pkl.gz', compression='gzip')
            df_train_titleonly.to_pickle(f'{path}{ds}_bert_cutT_titleonly_{seq_length_titleonly}.pkl.gz', compression='gzip')
            df_inspect.to_csv(f'{path}inspection/{ds}_bert_cutBTDS_{seq_length}.csv.gz', index=False)

        #################################################################

        Path('../../../data/processed/wdc-lspc/bert/inspection/').mkdir(parents=True, exist_ok=True)
        path = '../../../data/processed/wdc-lspc/bert/'

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        attributes = ['title', 'description', 'brand', 'specTableContent']
        seq_length = 512
        seq_length_titleonly = 128

        datasets_lspc_pretrain = ['pre-training_computers_only_new_15', 'pre-training_4cat_new_5']

        for ds in datasets_lspc_pretrain:
            df = pd.read_pickle(f'../../../data/raw/wdc-lspc/pre-training-set/{ds}.pkl.gz')
            df_train, df_train_titleonly, df_inspect = process_to_bert(df, attributes, tokenizer, seq_length,
                                                                       seq_length_titleonly, _att_to_seq_lspc, _cut_lspc)

            df_train.to_pickle(f'{path}{ds}_bert_cutBTDS_{seq_length}.pkl.gz', compression='gzip')
            df_train_titleonly.to_pickle(f'{path}{ds}_bert_cutT_titleonly_{seq_length_titleonly}.pkl.gz', compression='gzip')
            df_inspect.to_csv(f'{path}inspection/{ds}_bert_cutBTDS_{seq_length}.csv.gz', index=False)

    ################################################################

    if BUILD_PRETRAIN_COMPUTERS_AND_4CAT_MASKED:
        Path('../../../data/processed/wdc-lspc/bert/inspection/').mkdir(parents=True, exist_ok=True)
        path = '../../../data/processed/wdc-lspc/bert/'

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        attributes = ['title', 'description', 'brand', 'specTableContent']
        seq_length = 128
        seq_length_titleonly = 128

        datasets_lspc_pretrain = ['pre-training_computers_only_new_15', 'pre-training_4cat_new_5']

        for ds in datasets_lspc_pretrain:
            df = pd.read_pickle(f'../../../data/raw/wdc-lspc/pre-training-set/{ds}.pkl.gz')
            df_train, df_train_titleonly, df_inspect = process_to_bert(df, attributes, tokenizer, seq_length,
                                                                       seq_length_titleonly, _att_to_seq_lspc, _cut_lspc,
                                                                       mlm=True)

            df_train.to_pickle(f'{path}{ds}_bert_cutBTDS_MLM_{seq_length}.pkl.gz', compression='gzip')
            df_train_titleonly.to_pickle(f'{path}{ds}_bert_cutT_MLM_titleonly_{seq_length_titleonly}.pkl.gz',
                                         compression='gzip')
            df_inspect.to_csv(f'{path}inspection/{ds}_bert_cutBTDS_MLM_{seq_length}.csv.gz', index=False)

        #################################################################

        Path('../../../data/processed/wdc-lspc/bert/inspection/').mkdir(parents=True, exist_ok=True)
        path = '../../../data/processed/wdc-lspc/bert/'

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        attributes = ['title', 'description', 'brand', 'specTableContent']
        seq_length = 512
        seq_length_titleonly = 128

        datasets_lspc_pretrain = ['pre-training_computers_only_new_15', 'pre-training_4cat_new_5']

        for ds in datasets_lspc_pretrain:
            df = pd.read_pickle(f'../../../data/raw/wdc-lspc/pre-training-set/{ds}.pkl.gz')
            df_train, df_train_titleonly, df_inspect = process_to_bert(df, attributes, tokenizer, seq_length,
                                                                       seq_length_titleonly, _att_to_seq_lspc, _cut_lspc,
                                                                       mlm=True)

            df_train.to_pickle(f'{path}{ds}_bert_cutBTDS_MLM_{seq_length}.pkl.gz', compression='gzip')
            df_train_titleonly.to_pickle(f'{path}{ds}_bert_cutT_MLM_titleonly_{seq_length_titleonly}.pkl.gz',
                                         compression='gzip')
            df_inspect.to_csv(f'{path}inspection/{ds}_bert_cutBTDS_MLM_{seq_length}.csv.gz', index=False)