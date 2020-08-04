import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)
import os

from src.data import utils

PREPROCESS = True
# only needed to replicate construction of the intermediate training sets, which are provided in the download file
DEDUP = False

if __name__ == '__main__':

    print('PREPROCESSING CORPUS')

    corpus = pd.read_json('../../../data/raw/wdc-lspc/corpus/offers_corpus_english_v2_non_norm.json.gz', lines=True)

    # preprocess english corpus

    if PREPROCESS:
        print('BUILDING PREPROCESSED CORPUS...')
        corpus['title'] = corpus['title'].apply(utils.clean_string_wdcv2)
        corpus['description'] = corpus['description'].apply(utils.clean_string_wdcv2)
        corpus['brand'] = corpus['brand'].apply(utils.clean_string_wdcv2)
        corpus['price'] = corpus['price'].apply(utils.clean_string_wdcv2)
        corpus['specTableContent'] = corpus['specTableContent'].apply(utils.clean_specTableContent_wdcv2)

        os.makedirs(os.path.dirname('../../../data/interim/wdc-lspc/corpus/'), exist_ok=True)
        corpus.to_pickle('../../../data/interim/wdc-lspc/corpus/preprocessed_english_corpus.pkl.gz')
        print('FINISHED BUILDING PREPROCESSED CORPUS...')

    # build corpus deduplicated on title+description+brand+specTableContent
    if PREPROCESS and DEDUP:
        print('BUILDING DEDUPED CORPUS...')
        corpus_copy = corpus.copy()
        corpus_copy = corpus_copy.dropna(subset=['title'])
        corpus_copy = corpus_copy.fillna({'description':'', 'brand':'', 'specTableContent':''})

        corpus_copy['title+description+brand+specTableContent'] = corpus_copy['title'] + ' ' + corpus_copy['description'] + ' ' + corpus_copy['brand'] + ' ' + corpus_copy['specTableContent']
        corpus_copy['title+description+brand+specTableContent'] = corpus_copy['title+description+brand+specTableContent'].apply(lambda x: ' '.join(x.lower().split()))
        corpus_copy = corpus_copy.drop_duplicates(subset=['title+description+brand+specTableContent'])

        corpus_dedup = corpus.loc[corpus_copy.index]

        corpus_dedup.to_pickle('../../../data/interim/wdc-lspc/corpus/dedup_preprocessed_english_corpus.pkl.gz')
        print('FINISHED BUILDING DEDUPED CORPUS...')
