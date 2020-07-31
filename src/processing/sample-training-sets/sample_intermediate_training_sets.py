import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

from tqdm import tqdm
import pickle
import time

from gensim.corpora import Dictionary
from gensim.similarities import SparseMatrixSimilarity
from gensim.similarities import Similarity

# use cached files if code has at least been run once
USE_TEMP_CORPUS = False
USE_TEMP_POS_PAIRS = False

# select for which categories to build
CATEGORIES = ['Computers_and_Accessories']
#CATEGORIES = ['Camera_and_Photo', 'Shoes', 'Jewelry', 'Computers_and_Accessories']

# file naming handle
NAMING = 'computers_only'
#NAMING = '4cat'

def build_positive_pairs(corpus, clusters, attribute, num_pos):
    pos_pairs = []
    for current_cluster in tqdm(clusters):
        cluster_data = corpus[corpus['cluster_id'] == current_cluster]

        # build gensim dictionary, corpus and search index for selected cluster
        dct = Dictionary(cluster_data[attribute], prune_at=5000000)
        dct.filter_extremes(no_below=2, no_above=1.0, keep_n=None)
        gensim_corpus = [dct.doc2bow(text) for text in cluster_data[attribute]]
        index = SparseMatrixSimilarity(gensim_corpus, num_features=len(dct), num_best=80)

        # query up to 80 most similar offers, only offers with similarity > 0 will be returned
        query = index[gensim_corpus]

        for i, offer_sim_dup in enumerate(query):

            current_num_pos = num_pos
            current_id = cluster_data.iloc[i]['id']

            offer_sim = []

            # remove self
            for x in offer_sim_dup:
                if x[0] != i:
                    offer_sim.append(x)

            # check if any pairs > 0 similarity remain
            if len(offer_sim) == 0:
                pos_pairs.append((current_id, [[], []]))
                continue

            # adapt number of selectable pairs if too few available
            offer_len = len(offer_sim)
            if offer_len < current_num_pos:
                current_num_pos = offer_len

            if current_num_pos == 1:
                hard_pos = 1
                random_pos = 0
            elif current_num_pos % 2 == 1:
                hard_pos = int(current_num_pos / 2) + 1
                random_pos = int(current_num_pos / 2)
            else:
                hard_pos = int(current_num_pos / 2)
                random_pos = int(current_num_pos / 2)

            # get hard offers from bottom of list
            hard_offers = offer_sim[-hard_pos:]

            if random_pos == 0:
                pos_pairs.append((current_id, [[cluster_data.iloc[x[0]]['id'] for x in hard_offers], []]))
                continue

            # remaining offers
            rest = offer_sim[:-hard_pos]

            # randomly select from remaining
            random_select = random.sample(range(len(rest)), random_pos)
            random_offers = [rest[idx] for idx in random_select]

            hard_ids = [cluster_data.iloc[x[0]]['id'] for x in hard_offers]
            random_ids = [cluster_data.iloc[x[0]]['id'] for x in random_offers]

            pos_pairs.append((current_id, [hard_ids, random_ids]))
    return pos_pairs

def build_neg_pairs_for_cat(corpus, category, offers, attribute, num_neg):
    # select data from relevant category
    cat_data = corpus[corpus['category'] == category].copy()
    cat_data = cat_data.reset_index(drop=True)
    cat_data['subindex'] = list(cat_data.index)

    # build gensim dictionary, corpus and search index for selected cluster
    dct = Dictionary(cat_data[attribute], prune_at=5000000)
    dct.filter_extremes(no_below=2, no_above=0.8, keep_n=None)

    gensim_corpus = [dct.doc2bow(text) for text in cat_data[attribute]]

    index = Similarity(None, gensim_corpus, num_features=len(dct), num_best=200)

    # corpus to select negatives against
    corpus_neg_all = cat_data

    # corpus containing only offers for which negatives should be built
    corpus_neg = corpus_neg_all[corpus_neg_all['id'].isin(offers)]

    neg_pairs_cat = []

    query_corpus = [gensim_corpus[i] for i in list(corpus_neg['subindex'])]
    start = time.time()
    query = index[query_corpus]
    end = time.time()
    print(f'Category {category} query took {end - start} seconds')

    for i, offer_sim in enumerate(query):

        current_index = corpus_neg.iloc[i]['subindex']
        current_id = corpus_neg.iloc[i]['id']
        current_cluster_id = corpus_neg.iloc[i]['cluster_id']
        current_num_neg = num_neg

        # remove any offers with similarity 1.0
        sim_indices = []
        for x in offer_sim:
            if x[1] >= 1.0:
                continue
            else:
                sim_indices.append(x[0])

        possible_pairs = corpus_neg_all.loc[sim_indices]

        # filter by cluster_id, i.e. only 1 offer per cluster remains to allow for product diversity
        idx = sorted(np.unique(possible_pairs['cluster_id'], return_index=True)[1])

        possible_pairs = possible_pairs.iloc[idx]

        # remove any offer from same cluster
        possible_pairs = possible_pairs[possible_pairs['cluster_id'] != current_cluster_id]

        possible_pairs_len = len(possible_pairs)

        # check if any pairs > 0 similarity remain
        if possible_pairs_len == 0:
            neg_pairs_cat.append((current_id, [[], []]))
            continue

        # adapt number of selectable pairs if too few available
        if possible_pairs_len < current_num_neg:
            current_num_neg = possible_pairs_len

        if current_num_neg == 1:
            hard_neg = 1
            random_neg = 0
        elif current_num_neg % 2 == 1:
            hard_neg = int(current_num_neg / 2) + 1
            random_neg = int(current_num_neg / 2)
        else:
            hard_neg = int(current_num_neg / 2)
            random_neg = int(current_num_neg / 2)

        # select hard pairs from top of list
        candidates = possible_pairs.iloc[:hard_neg]

        hard_pairs = candidates['id'].tolist()

        if random_neg == 0:
            neg_pairs_cat.append((current_id, [hard_pairs, []]))
            continue
        else:
            remove = list(candidates.index)
            remove.append(current_index)

            # randomly select from all offers among same category
            random_select = random.sample(range(len(corpus_neg_all)), random_neg)
            random_pairs = corpus_neg_all.iloc[random_select]
            while (any(random_pairs['id'].isin(remove)) or any(random_pairs['cluster_id'] == current_cluster_id)):
                random_select = random.sample(range(len(corpus_neg_all)), random_neg)
                random_pairs = corpus_neg_all.iloc[random_select]
            random_pairs = random_pairs['id'].tolist()

            combined_pairs = [hard_pairs, random_pairs]
        neg_pairs_cat.append((current_id, combined_pairs))

    return neg_pairs_cat

if __name__ == '__main__':
    if not USE_TEMP_CORPUS:
        corpus = pd.read_pickle('../../../data/interim/wdc-lspc/corpus/dedup_preprocessed_english_corpus.pkl.gz')
        corpus = corpus.fillna('')

        corpus['desc5'] = corpus['description'].apply(lambda x: ' '.join(x.split()[:5]))

        corpus['title_for_ts'] = corpus['title'] + ' ' + corpus['desc5']
        corpus = corpus.drop(['identifiers', 'title', 'description', 'brand', 'price', 'keyValuePairs', 'specTableContent', 'desc5'], axis=1)

        corpus['index'] = list(corpus.index)
        corpus['title_for_ts'] = corpus['title_for_ts'].str.lower()
        corpus['title_for_ts'] = corpus['title_for_ts'].str.split()
        corpus['title_for_ts'] = corpus['title_for_ts'].apply(lambda x: list(set(x)))
        corpus.to_pickle('../../../data/interim/wdc-lspc/corpus/temp.pkl.gz')
    else:
        corpus = pd.read_pickle('../../../data/interim/wdc-lspc/corpus/temp.pkl.gz')

    # remove GS clusters
    gs = pd.read_json('../../../data/raw/wdc-lspc/gold-standards/all_gs.json.gz', lines=True)
    clusters1 = gs['cluster_id_left'].tolist()
    clusters2 = gs['cluster_id_right'].tolist()
    clusters1.extend(clusters2)
    gs_clusters = list(set(clusters1))
    corpus = corpus[~corpus['cluster_id'].isin(gs_clusters)]

    # remove clusters from new GS
    gs_new = pd.read_json('../../../data/raw/wdc-lspc/gold-standards/computers_new_testset_500.json.gz', lines=True)
    clusters1 = gs_new['cluster_id_left'].tolist()
    clusters2 = gs_new['cluster_id_right'].tolist()
    clusters1.extend(clusters2)
    gs_new_clusters = list(set(clusters1))
    corpus = corpus[~corpus['cluster_id'].isin(gs_new_clusters)]

    categories = CATEGORIES

    corpus = corpus[corpus['category'].isin(categories)]
    categories = list(corpus['category'].unique())
    print(f'Categories used for building: {categories}')

    # remove clusters with only 1 offer for positives
    gt1_bool = corpus['cluster_id'].value_counts() > 1
    clusters_gt1 = list(gt1_bool[gt1_bool == True].index)
    corpus_pos = corpus[corpus['cluster_id'].isin(clusters_gt1)]

    # remove clusters with more than 80 offers for positives
    lt80_bool = corpus_pos['cluster_id'].value_counts() <= 80
    clusters_lt80 = list(lt80_bool[lt80_bool == True].index)
    corpus_pos = corpus_pos[corpus_pos['cluster_id'].isin(clusters_lt80)]

    # Max amount of pairs to build per offer, leave as is, downsampling occurs later
    X = 20
    clusters_pos = corpus_pos['cluster_id'].unique()
    num_pos = X
    attribute = 'title_for_ts'

    if not USE_TEMP_POS_PAIRS:
        pos_pairs = build_positive_pairs(corpus_pos, clusters_pos, attribute, num_pos)

        pos_pairs_dedup = [x for x in pos_pairs if len(x[1][0]) + len(x[1][1]) > 0]
        pos_pairs = pos_pairs_dedup

        with open(f'../../../data/interim/wdc-lspc/corpus/positive_pairs_{NAMING}_new_{X}.pkl', 'wb') as f:
            # store the data as binary data stream
            pickle.dump(pos_pairs, f)
    else:
        with open(f'../../../data/interim/wdc-lspc/corpus/positive_pairs_{NAMING}_new_{X}.pkl', 'rb') as f:
            # store the data as binary data stream
            pos_pairs = pickle.load(f)

    # get counts of positive pairs per offer to build same amount of negs
    neg_counts = {x[0]:len(x[1][0])+len(x[1][1]) for x in tqdm(pos_pairs)}

    num_neg = X

    offers = list(neg_counts.keys())
    attribute = 'title_for_ts'
    for category in tqdm(categories):
        neg_pairs = build_neg_pairs_for_cat(corpus, category, offers, attribute, num_neg)
        with open(f'../../../data/interim/wdc-lspc/corpus/negative_pairs_{category}_{NAMING}_new_{X}.pkl', 'wb') as f:
            # store the data as binary data stream
            pickle.dump(neg_pairs, f)
