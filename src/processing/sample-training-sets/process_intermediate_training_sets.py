import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

from tqdm import tqdm
import pickle
from pathlib import Path

# select categories
CATEGORIES = ['Computers_and_Accessories']
#CATEGORIES = ['Camera_and_Photo', 'Shoes', 'Jewelry', 'Computers_and_Accessories']

# file naming handle
NAMING = 'computers_only'
#NAMING = '4cat'

def subselect_pairs(full_pairs, amount):
    start_amount = amount
    new_pairs = []
    for pairs in tqdm(full_pairs):
        amount = start_amount
        pairs_len = len(pairs[1][0]) + len(pairs[1][1])

        if pairs_len < amount:
            amount = pairs_len

        if amount == 1:
            hard_pairs = 1
            random_pairs = 0
        elif amount % 2 == 1:
            hard_pairs = int(amount / 2) + 1
            random_pairs = int(amount / 2)
        else:
            hard_pairs = int(amount / 2)
            random_pairs = int(amount / 2)

        new_pairs.append((pairs[0], [pairs[1][0][:hard_pairs], pairs[1][1][:random_pairs]]))
    return new_pairs

def build_deduped_training_set(pos_pairs, neg_pairs, amount, corpus):
    ids_left = []
    full_pos_pairs = []
    for pair in pos_pairs:
        ids_left.append(pair[0])
        for i in range(len(pair[1][0])):
            full_pos_pairs.append(f'{str(pair[0])}#{str(pair[1][0][i])}')
        for i in range(len(pair[1][1])):
            full_pos_pairs.append(f'{str(pair[0])}#{str(pair[1][1][i])}')

    full_neg_pairs = []
    for pair in neg_pairs:
        for i in range(len(pair[1][0])):
            full_neg_pairs.append(f'{str(pair[0])}#{str(pair[1][0][i])}')
        for i in range(len(pair[1][1])):
            full_neg_pairs.append(f'{str(pair[0])}#{str(pair[1][1][i])}')

    dedup_full_pos = list(set(full_pos_pairs))
    dedup_full_neg = list(set(full_neg_pairs))

    corpus_selection = corpus.loc[ids_left]
    clusters = len(corpus_selection['cluster_id'].unique())
    print(f'Amount of positives and negatives selected for each offer: up to {amount} each')
    print(f'Amount of offers with positives and negatives:{len(pos_pairs)}')
    print(f'Selected from {clusters} clusters')
    print(f'Full Amount of Positives: {len(dedup_full_pos)}')
    print(f'Full Amount of Negatives: {len(dedup_full_neg)}')

    super_dedup_full_pos = set()
    super_dedup_full_neg = set()

    for pair in dedup_full_pos:
        split = pair.split('#')
        reversed_pair = f'{split[1]}#{split[0]}'
        if reversed_pair not in super_dedup_full_pos:
            super_dedup_full_pos.add(pair)

    for pair in dedup_full_neg:
        split = pair.split('#')
        reversed_pair = f'{split[1]}#{split[0]}'
        if reversed_pair not in super_dedup_full_neg:
            super_dedup_full_neg.add(pair)

    print(f'Deduped Amount of Positives: {len(super_dedup_full_pos)}')
    print(f'Deduped Amount of Negatives: {len(super_dedup_full_neg)}')

    super_dedup_full_pos = list(super_dedup_full_pos)
    super_dedup_full_neg = list(super_dedup_full_neg)

    pos_set = [[x, 1] for x in super_dedup_full_pos]
    neg_set = [[x, 0] for x in super_dedup_full_neg]

    combined_trainingset = pos_set + neg_set

    trainingset_df = pd.DataFrame(combined_trainingset, columns=['pair_id', 'label'])

    return trainingset_df

if __name__ == '__main__':
    corpus = pd.read_pickle('../../../data/interim/wdc-lspc/corpus/preprocessed_english_corpus.pkl.gz')
    corpus = corpus.set_index('id', drop=False)

    categories = CATEGORIES

    neg_pairs = []
    for category in categories:
        with open(f'../../../data/interim/wdc-lspc/corpus/negative_pairs_{category}_{NAMING}_new_20.pkl', 'rb') as f:
            neg_pairs_cat = pickle.load(f)
        neg_pairs.extend(neg_pairs_cat)

    with open(f'../../../data/interim/wdc-lspc/corpus/positive_pairs_{NAMING}_new_20.pkl', 'rb') as f:
        pos_pairs = pickle.load(f)

    amount = 20
    # sample certain amount of pairs only
    subselect = True

    if subselect:
        if NAMING == 'computers_only':
            amount = 15
        elif NAMING == '4cat':
            amount = 5
        else:
            amount = 20
        pos_pairs = subselect_pairs(pos_pairs, amount)
        neg_pairs = subselect_pairs(neg_pairs, amount)

        for pairs in pos_pairs:
            assert len(pairs[1][0]) + len(pairs[1][1]) <= amount

        for pairs in neg_pairs:
            assert len(pairs[1][0]) + len(pairs[1][1]) <= amount

    result = build_deduped_training_set(pos_pairs, neg_pairs, amount, corpus)

    Path('../../../data/raw/wdc-lspc/pre-training-set').mkdir(parents=True, exist_ok=True)

    result.to_csv(f'../../../data/raw/wdc-lspc/pre-training-set/pre_training_{NAMING}_new_{amount}.csv.gz', index=False)
