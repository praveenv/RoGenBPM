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
    env4_X = []
    env4_Y = []

    df = pd.read_csv('../data/BPI_Challenge_2018_Normal.csv')
    unique_case_id = df['case:concept:name'].unique()

    activity_list = df['activity'].unique()
    
    for idx, act in enumerate(tqdm(activity_list, desc='activity')):
        print(idx, act)
        df.loc[df.activity == act, 'activity'] = idx + 1

    padded_ngrams = partial(ngrams, pad_left=True, pad_right=False, left_pad_symbol=0, right_pad_symbol=0)
    float_padded_ngrams = partial(ngrams, pad_left=True, pad_right=False, left_pad_symbol=0.0, right_pad_symbol=0.0)

    count = 0
    for case in tqdm(unique_case_id):
        subset_df = df[df['case:concept:name'] == case]
        subset_df.reset_index(inplace=True, drop=True)

        current_environment = subset_df['department'][0]

        activity_list = list(subset_df['activity'])

        subset_df['time:timestamp'] = [datetime.fromisoformat(timestamp) for timestamp in subset_df['time:timestamp']]
        subset_df['time_delta'] = subset_df['time:timestamp'].diff()
        subset_df['time_delta'] = subset_df['time_delta'] / timedelta(days=1)
        subset_df['time_delta'][0] = 0

        timestamps = list(subset_df['time_delta'])
        young_farmer = list(subset_df['case:young farmer'])
        small_farmer = list(subset_df['case:small farmer'])
        risk = list(subset_df['case:selected_risk'])

        activity_X, activity_Y = generatePadding(False, activity_list, MAX_NGRAM_SIZE)
        timestamp_X, timestamp_Y = generatePadding(True, timestamps, MAX_NGRAM_SIZE)
        young_farmer_X, young_farmer_Y = generatePadding(False, young_farmer, MAX_NGRAM_SIZE)
        small_farmer_X, small_farmer_Y = generatePadding(False, small_farmer, MAX_NGRAM_SIZE)
        risk_X, risk_Y = generatePadding(False, risk, MAX_NGRAM_SIZE)

        for i in range(len(activity_X)):
            activity_ngram = activity_X[i]
            timestamp_ngram = timestamp_X[i]
            young_farmer_ngram = young_farmer_X[i]
            small_farmer_ngram = small_farmer_X[i]
            risk_ngram = risk_X[i]
            
            for j in range(MAX_NGRAM_SIZE):
                if current_environment == 1:
                    env1_X.append([activity_ngram[j], timestamp_ngram[j], young_farmer_ngram[j], small_farmer_ngram[j], risk_ngram[j]])
                elif current_environment == 2:
                    env2_X.append([activity_ngram[j], timestamp_ngram[j], young_farmer_ngram[j], small_farmer_ngram[j], risk_ngram[j]])
                elif current_environment == 3:
                    env3_X.append([activity_ngram[j], timestamp_ngram[j], young_farmer_ngram[j], small_farmer_ngram[j], risk_ngram[j]])
                elif current_environment == 4:
                    env4_X.append([activity_ngram[j], timestamp_ngram[j], young_farmer_ngram[j], small_farmer_ngram[j], risk_ngram[j]])

        for j in range(len(activity_Y)):
            if current_environment == 1:
                env1_Y.append([activity_Y[j], timestamp_Y[j]])
            elif current_environment == 2:
                env2_Y.append([activity_Y[j], timestamp_Y[j]])
            elif current_environment == 3:
                env3_Y.append([activity_Y[j], timestamp_Y[j]])
            elif current_environment == 4:
                env4_Y.append([activity_Y[j], timestamp_Y[j]])    
    

    df_X = pd.DataFrame(env1_X, columns=['activity', 'timestamp', 'young_farmer','small_farmer','risk'])
    df_X.to_csv('../data/BPI18_env1_X.csv', index=False)
    df_Y = pd.DataFrame(env1_Y, columns=['activity', 'timestamp'])
    df_Y.to_csv('../data/BPI18_env1_Y.csv', index=False)

    df_X = pd.DataFrame(env2_X, columns=['activity', 'timestamp', 'young_farmer','small_farmer','risk'])
    df_X.to_csv('../data/BPI18_env2_X.csv', index=False)
    df_Y = pd.DataFrame(env2_Y, columns=['activity', 'timestamp'])
    df_Y.to_csv('../data/BPI18_env2_Y.csv', index=False)

    df_X = pd.DataFrame(env3_X, columns=['activity', 'timestamp', 'young_farmer','small_farmer','risk'])
    df_X.to_csv('../data/BPI18_env3_X.csv', index=False)
    df_Y = pd.DataFrame(env3_Y, columns=['activity', 'timestamp'])
    df_Y.to_csv('../data/BPI18_env3_Y.csv', index=False)

    df_X = pd.DataFrame(env4_X, columns=['activity', 'timestamp', 'young_farmer','small_farmer','risk'])
    df_X.to_csv('../data/BPI18_env4_X.csv', index=False)
    df_Y = pd.DataFrame(env4_Y, columns=['activity', 'timestamp'])
    df_Y.to_csv('../data/BPI18_env4_Y.csv', index=False)

def main():
    generateNgram()

if __name__ == '__main__':
    main()