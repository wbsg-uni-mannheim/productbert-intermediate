import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

from tqdm import tqdm

# select categories
CATEGORIES = ['Computers_and_Accessories']
# CATEGORIES = ['Camera_and_Photo', 'Shoes', 'Jewelry', 'Computers_and_Accessories']

# file naming handle
NAMING = 'computers_only'
# NAMING = '4cat'

# amount of selected pairs during processing stage
if NAMING == 'computers_only':
    AMOUNT = 15
elif NAMING == '4cat':
    AMOUNT = 5
else:
    AMOUNT = 20

if __name__ == '__main__':
    pair_ids = pd.read_csv(f'../../../data/raw/wdc-lspc/pre-training-set/pre_training_{NAMING}_new_{AMOUNT}.csv.gz')

    corpus = pd.read_pickle('../../../data/interim/wdc-lspc/corpus/preprocessed_english_corpus.pkl.gz')
    corpus = corpus.set_index('id', drop=False)

    gs = pd.read_pickle('../../../data/interim/wdc-lspc/gold-standards/preprocessed_all_gs.pkl.gz')

    gs_pairs = gs['pair_id'].tolist()

    left_ids = []
    right_ids = []
    labels = []
    combined_ids = []

    # double-check GS inclusion and randomize left and right offer of pair
    for i, pair_id, label in tqdm(pair_ids.itertuples(), total=len(pair_ids)):
        split = pair_id.split('#')
        if pair_id not in gs_pairs and f'{split[1]}#{split[0]}' not in gs_pairs:
            random_pick = random.randint(0,1)
            if random_pick == 1:
                left_ids.append(int(split[0]))
                right_ids.append(int(split[1]))
                combined_ids.append(split[0]+'#'+split[1])
            else:
                left_ids.append(int(split[1]))
                right_ids.append(int(split[0]))
                combined_ids.append(split[1]+'#'+split[0])
            labels.append(label)
        else:
            print('This cannot happen!')

    left_df = corpus.loc[left_ids]
    right_df = corpus.loc[right_ids]

    left_df = left_df.reset_index(drop=True)
    right_df = right_df.reset_index(drop=True)

    final_training_set = left_df.join(right_df, lsuffix='_left', rsuffix='_right')
    final_training_set['label'] = labels
    final_training_set['pair_id'] = combined_ids

    print(f'Size of pre-training set: {len(final_training_set)}')
    print(f'Distribution of training set labels: \n{final_training_set["label"].value_counts()}')

    final_training_set.to_pickle(f'../../../data/raw/wdc-lspc/pre-training-set/pre_training_{NAMING}_new_5.pkl.gz')
