'''
json -> pd -> pkl
'''

import pickle
import pandas as pd


def to_df(file_path):
    '''
    转化为DataFrame结构
    :param file_path: 文件路径
    :return:
    '''

    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df


reviews_path = '../data/reviews_Electronics_5.json'
reviews_df = to_df(reviews_path)

reviews_pkl = '../data/reviews.pkl'
with open(reviews_pkl, 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

meta_path = '../data/meta_Electronics.json'
meta_df = to_df(meta_path)
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)

meta_pkl = '../data/meta.pkl'
with open(meta_pkl, 'wb') as f:
    pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)

