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

    df = pd.read_csv('../data/BPI_Challenge_2013_Normal.csv')
    unique_case_id = df['case:concept:name'].unique()

    activity_list = df['Activity'].unique()
    resource_list = df['Resource'].unique()
    
    for idx, act in enumerate(tqdm(activity_list, desc='activity')):
        df.loc[df['Activity'] == act, 'Activity'] = idx + 1

    for idx, act in enumerate(tqdm(resource_list, desc='resource')):
        df.loc[df['Resource'] == act, 'Resource'] = idx + 1

    df['Resource'] = df['Resource'] / df['Resource'].max()
    df['case:variant-index'] = df['case:variant-index'] / df['case:variant-index'].max()

    padded_ngrams = partial(ngrams, pad_left=True, pad_right=False, left_pad_symbol=0, right_pad_symbol=0)
    float_padded_ngrams = partial(ngrams, pad_left=True, pad_right=False, left_pad_symbol=0.0, right_pad_symbol=0.0)

    count = 0
    for case in tqdm(unique_case_id):
        subset_df = df[df['case:concept:name'] == case]
        subset_df.reset_index(inplace=True, drop=True)
        current_environment = subset_df['environment'][0]

        activity_list = list(subset_df['Activity'])

        timestamps = list(subset_df['time_delta'])
        resource_list = list(subset_df['Resource'])
        amount_list = list(subset_df['case:variant-index'])

        activity_X, activity_Y = generatePadding(False, activity_list, MAX_NGRAM_SIZE)
        timestamp_X, timestamp_Y = generatePadding(True, timestamps, MAX_NGRAM_SIZE)
        resource_X , resource_Y = generatePadding(False, resource_list, MAX_NGRAM_SIZE)
        amount_X , amount_Y = generatePadding(False, amount_list, MAX_NGRAM_SIZE)

        for i in range(len(activity_X)):
            activity_ngram = activity_X[i]
            timestamp_ngram = timestamp_X[i]
            resource_ngram = resource_X[i]
            amount_ngram = amount_X[i]
            
            for j in range(MAX_NGRAM_SIZE):
                if current_environment == 1:
                    env1_X.append([activity_ngram[j], timestamp_ngram[j], resource_ngram[j], amount_ngram[j]])
                elif current_environment == 2:
                    env2_X.append([activity_ngram[j], timestamp_ngram[j], resource_ngram[j], amount_ngram[j]])
                else:
                    env3_X.append([activity_ngram[j], timestamp_ngram[j], resource_ngram[j], amount_ngram[j]])

        for j in range(len(activity_Y)):
            if current_environment == 1:
                env1_Y.append([activity_Y[j], timestamp_Y[j]])
            elif current_environment == 2:
                env2_Y.append([activity_Y[j], timestamp_Y[j]])
            else:
                env3_Y.append([activity_Y[j], timestamp_Y[j]])
    

    df_X = pd.DataFrame(env1_X, columns=['activity', 'timestamp', 'resource', 'variant'])
    df_X.to_csv('../data/Helpdesk_env1_X.csv', index=False)
    df_Y = pd.DataFrame(env1_Y, columns=['activity', 'timestamp'])
    df_Y.to_csv('../data/Helpdesk_env1_Y.csv', index=False)

    df_X = pd.DataFrame(env2_X, columns=['activity', 'timestamp', 'resource', 'variant'])
    df_X.to_csv('../data/Helpdesk_env2_X.csv', index=False)
    df_Y = pd.DataFrame(env2_Y, columns=['activity', 'timestamp'])
    df_Y.to_csv('../data/Helpdesk_env2_Y.csv', index=False)

    df_X = pd.DataFrame(env3_X, columns=['activity', 'timestamp', 'resource', 'variant'])
    df_X.to_csv('../data/Helpdesk_env3_X.csv', index=False)
    df_Y = pd.DataFrame(env3_Y, columns=['activity', 'timestamp'])
    df_Y.to_csv('../data/Helpdesk_env3_Y.csv', index=False)


def main():
    generateNgram()

if __name__ == '__main__':
    main()