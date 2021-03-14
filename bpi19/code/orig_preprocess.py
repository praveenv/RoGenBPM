import pandas as pd
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
import collections
import math
import random
import numpy as np
from sklearn.utils import shuffle
from collections import Counter
from functools import partial
from nltk import ngrams
pd.options.mode.chained_assignment = None  # default='warn'

random.seed(10)

def generatePadding(float_bool, padding_list, ngram_size):

    padded_ngrams = partial(ngrams, pad_left=True, pad_right=False, left_pad_symbol=0, right_pad_symbol=0)
    float_padded_ngrams = partial(ngrams, pad_left=True, pad_right=False, left_pad_symbol=0.0, right_pad_symbol=0.0)

    if float_bool == False:
        X = list(padded_ngrams(padding_list,ngram_size))
        Y = [x[-1] for x in X]
        X = list(X[:-1])
        Y = Y[1:]
    else:
        X = list(float_padded_ngrams(padding_list,ngram_size))
        Y = [x[-1] for x in X]
        X = list(X[:-1])
        Y = Y[1:]

    return X, Y


def generateNgram():

    MAX_NGRAM_SIZE = 10

    env1_X = []
    env1_Y = []
    env2_X = []
    env2_Y = []
    env3_X = []
    env3_Y = []

    df = pd.read_csv('../data/BPI_2019_Normal.csv')
    unique_case_id = df['case:concept:name'].unique()
    
    activity_list = df['concept:name'].unique()

    for idx, act in enumerate(tqdm(activity_list, desc='activity')):
        df.loc[df['concept:name'] == act, 'concept:name'] = idx + 1

    padded_ngrams = partial(ngrams, pad_left=True, pad_right=False, left_pad_symbol=0, right_pad_symbol=0)
    float_padded_ngrams = partial(ngrams, pad_left=True, pad_right=False, left_pad_symbol=0.0, right_pad_symbol=0.0)

    count = 0
    for case in tqdm(unique_case_id):
        subset_df = df[df['case:concept:name'] == case]
        subset_df.reset_index(inplace=True, drop=True)

        current_environment = subset_df['environment'][0]

        activity_list = list(subset_df['concept:name'])

        subset_df['time:timestamp'] = [datetime.fromisoformat(timestamp) for timestamp in subset_df['time:timestamp']]
        subset_df['time_delta'] = subset_df['time:timestamp'].diff()
        subset_df['time_delta'] = subset_df['time_delta'] / timedelta(days=1)
        subset_df['time_delta'][0] = 0

        timestamps = list(subset_df['time_delta'])
        spend_area = list(subset_df['case:Spend area text'])
        doctype = list(subset_df['case:Document Type'])
        itemtype = list(subset_df['case:Item Type'])
        itemcat = list(subset_df['case:Item Category'])
        inv_flag = list(subset_df['case:GR-Based Inv. Verif.'])
        gr_flag = list(subset_df['case:Goods Receipt'])

        activity_X, activity_Y = generatePadding(False, activity_list, MAX_NGRAM_SIZE)
        timestamp_X, timestamp_Y = generatePadding(True, timestamps, MAX_NGRAM_SIZE)
        spend_area_X, spend_area_Y = generatePadding(False, spend_area, MAX_NGRAM_SIZE)
        doctype_X, doctype_Y = generatePadding(False, doctype, MAX_NGRAM_SIZE)
        itemtype_X, itemtype_Y = generatePadding(False, itemtype, MAX_NGRAM_SIZE)
        itemcat_X, itemcat_Y = generatePadding(False, itemcat, MAX_NGRAM_SIZE)
        inv_flag_X, inv_flag_Y = generatePadding(False, inv_flag, MAX_NGRAM_SIZE)
        gr_flag_X, gr_flag_Y = generatePadding(False,  gr_flag, MAX_NGRAM_SIZE)

        for i in range(len(activity_X)):
            activity_ngram = activity_X[i]
            timestamp_ngram = timestamp_X[i]
            spend_area_ngram = spend_area_X[i]
            doctype_ngram = doctype_X[i]
            itemtype_ngram = itemtype_X[i]
            itemcat_ngram = itemcat_X[i]
            inv_flag_ngram = inv_flag_X[i]
            gr_flag_ngram = gr_flag_X[i]
            
            for j in range(MAX_NGRAM_SIZE):
                if current_environment == 1:
                    env1_X.append([activity_ngram[j], timestamp_ngram[j], spend_area_ngram[j], doctype_ngram[j], itemtype_ngram[j],
                    itemcat_ngram[j], inv_flag_ngram[j], gr_flag_ngram[j]])
                elif current_environment == 2:
                    env2_X.append([activity_ngram[j], timestamp_ngram[j], spend_area_ngram[j], doctype_ngram[j], itemtype_ngram[j],
                    itemcat_ngram[j], inv_flag_ngram[j], gr_flag_ngram[j]])
                elif current_environment == 3:
                    env3_X.append([activity_ngram[j], timestamp_ngram[j], spend_area_ngram[j], doctype_ngram[j], itemtype_ngram[j],
                    itemcat_ngram[j], inv_flag_ngram[j], gr_flag_ngram[j]])

        for j in range(len(activity_Y)):
            if current_environment == 1:
                env1_Y.append([activity_Y[j], timestamp_Y[j]])
            elif current_environment == 2:
                env2_Y.append([activity_Y[j], timestamp_Y[j]])
            elif current_environment == 3:
                env3_Y.append([activity_Y[j], timestamp_Y[j]])
    

    df_X = pd.DataFrame(env1_X, columns=['activity', 'timestamp', 'spend_area','doctype', 'itemtype','item_category','inv_verif_flag', 'goods_receipt_flag'])
    df_X.to_csv('../data/BPI19_env1_X.csv', index=False)
    df_Y = pd.DataFrame(env1_Y, columns=['activity', 'timestamp'])
    df_Y.to_csv('../data/BPI19_env1_Y.csv', index=False)

    df_X = pd.DataFrame(env2_X, columns=['activity', 'timestamp', 'spend_area','doctype', 'itemtype','item_category','inv_verif_flag', 'goods_receipt_flag'])
    df_X.to_csv('../data/BPI19_env2_X.csv', index=False)
    df_Y = pd.DataFrame(env2_Y, columns=['activity', 'timestamp'])
    df_Y.to_csv('../data/BPI19_env2_Y.csv', index=False)

    df_X = pd.DataFrame(env3_X, columns=['activity', 'timestamp', 'spend_area','doctype', 'itemtype','item_category','inv_verif_flag', 'goods_receipt_flag'])
    df_X.to_csv('../data/BPI19_env3_X.csv', index=False)
    df_Y = pd.DataFrame(env3_Y, columns=['activity', 'timestamp'])
    df_Y.to_csv('../data/BPI19_env3_Y.csv', index=False)



def main():
    generateNgram()

if __name__ == '__main__':
    main()