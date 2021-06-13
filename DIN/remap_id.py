'''
reviews_df保留'reviewerID【用户ID】, 'asin'【产品ID】, 'unixReviewTime'【浏览时间】三列
meta_df保留'asin'【产品ID】, 'categories'【种类】两列

reviews.pkl: 1689188 * 9
['reviewerID', 'asin', 'reviewerName', 'helpful', 'reviewText',
       'overall', 'summary', 'unixReviewTime', 'reviewTime']

       reviewerID        asin  ... unixReviewTime   reviewTime
0   AO94DHGC771SJ  0528881469  ...     1370131200   06 2, 2013
1   AMO214LNFCEI4  0528881469  ...     1290643200  11 25, 2010

meta.pkl: 63001 * 9
['asin', 'imUrl', 'description', 'categories', 'title', 'price',
       'salesRank', 'related', 'brand']
         asin                                         categories
0  0528881469  [[Electronics, GPS & Navigation, Vehicle GPS, ...

'''

import random
import pickle
import numpy as np
import pandas as pd


random.seed(2021)


def build_map(df, col_name):
    '''
    制作一个映射，键为列名，值为序列数字
    :param df: reviews_df / meta_df
    :param col_name: 列名
    :return: 字典，键
    '''

    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


# reviews
reviews_pkl = '../data/reviews.pkl'
reviews_df = pd.read_pickle(reviews_pkl)
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

# meta
meta_pkl = '../data/meta.pkl'
meta_df = pd.read_pickle(meta_pkl)
meta_df = meta_df[['asin', 'categories']]
# 类别只保留最后一个
meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

# meta_df文件的物品ID映射
asin_map, asin_key = build_map(meta_df, 'asin') # map: {'0528881469': 0, '0594451647': 1, '0594481813': 2, '0972683275': 3, ...}
# meta_df文件物品种类映射
cate_map, cate_key = build_map(meta_df, 'categories')   # map: {'3D Glasses': 0, 'AC Adapters': 1, 'APS Cameras': 2, 'AV Receivers & Amplifiers': 3, ...}
# reviews_df文件用户ID映射
revi_map, revi_key = build_map(reviews_df, 'reviewerID')# map: {'A000715434M800HLCENK9': 0, 'A00101847G3FJTWYGNQA': 1, 'A00166281YWM98A3SVD55': 2, ...}

# user_count: 192403	item_count: 63001	cate_count: 801	example_count: 1689188
user_count, item_count, cate_count, example_count = \
    len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]

# 按物品id排序，并重置索引
meta_df = meta_df.sort_values('asin')
meta_df = meta_df.reset_index(drop=True)

# reviews_df文件物品id进行映射，并按照用户id、浏览时间进行排序，重置索引
reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
reviews_df = reviews_df.reset_index(drop=True)
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

# 各个物品对应的类别
cate_list = np.array(meta_df['categories'], dtype='int32')

# 保存所需数据为pkl文件

remap_pkl = '../data/remap.pkl'
with open(remap_pkl, 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count, example_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)
