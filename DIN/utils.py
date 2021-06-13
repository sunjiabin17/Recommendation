'''
create amazon electronic dataset
'''
import pandas as pd
import numpy as np
import pickle
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sparseFeature(feat, feat_num, embed_dim=4):
    '''
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    '''
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    '''
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    '''
    return {'feat': feat}


def create_amazon_electronic_dataset(file, embed_dim=8, maxlen=40):
    '''

    :param file: dataset path
    :param embed_dim: latent factor
    :param maxlen:
    :return: user_num, item_num, train_df, test_df
    '''

    print('==========Data Preprocess Start===========')
    pkl = '../data/remap.pkl'
    with open(pkl, 'rb') as f:
        reviews_df = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, example_count = pickle.load(f)

    reviews_df = reviews_df
    reviews_df.columns = ['user_id', 'item_id', 'time']
    train_data, val_data, test_data = [], [], []

    for user_id, hist in tqdm(reviews_df.groupby('user_id')):
        pos_list = hist['item_id'].tolist() # 用户user_id看过的item

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))]    # 用户user_id没看过的item
        hist = []
        for i in range(1, len(pos_list)):
            hist.append([pos_list[i-1], cate_list[pos_list[i-1]]])  # [物品item， item对应的种类]
            hist_i = hist.copy()
            '''
            数据集解释
            hist_i: 用户之前浏览过的商品集合
            [pos_list[i], cate_list[pos_list[i]]]: 用户下一个浏览的商品及种类（正例）
            [neg_list[i], cate_list[neg_list[i]]]: 用户下一个没浏览过的商品（作为反例）
            label: 标记正反例
            '''
            if i == len(pos_list) - 1:  # 每个用户看过的item，最后一个作为测试集
                test_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                test_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
            elif i == len(pos_list) - 2:# 每个用户看过的item，倒数第二个作为验证集
                val_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                val_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
            else:                       # 每个用户看过的item，其他作为训练集
                train_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                train_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])

    feature_columns = [[], [sparseFeature('item_id', item_count, embed_dim),]]
    behavior_list = ['item_id']

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['hist', 'target_item', 'label'])

    print('==========Padding==========')
    train_X = [np.array([0.] * len(train)),
               np.array([0] * len(train)),
               pad_sequences(train['hist'], maxlen=maxlen),
               np.array(train['target_item'].tolist())]
    train_y = train['label'].values

    val_X = [np.array([0.] * len(val)),
             np.array([0] * len(val)),
               pad_sequences(val['hist'], maxlen=maxlen),
               np.array(val['target_item'].tolist())]
    val_y = val['label'].values

    test_X = [np.array([0.] * len(test)),
              np.array([0] * len(test)),
               pad_sequences(test['hist'], maxlen=maxlen),
               np.array(test['target_item'].tolist())]
    test_y = test['label'].values

    print('==========Data Preprocess Done==========')
    return feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)

if __name__ == '__main__':
    create_amazon_electronic_dataset('../data/remap.pkl')
