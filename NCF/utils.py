import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
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

def create_ml1m_dataset(file, trans_score=2, embed_dim=8, test_neg_num=100):
    '''
    :param file: dataset path
    :param trans_score: a scalar. Greater than it is 1, and less than it is 0
    :param embed_dim: a scalar. latent factor
    :param test_neg_num: a scalar. the number of test negative samples
    :return: user_num, item_num, train_df, test_df
    '''
    print('========data preprocess start========')
    data_df = pd.read_csv(file, sep='::', engine='python',
                          names=['user_id', 'item_id', 'label', 'timestamp'])   # (1000209, 4)
    # filtering
    data_df['item_count'] = data_df.groupby('item_id')['item_id'].transform('count')    # (1000209, 5)
    data_df = data_df[data_df.item_count >= 5]      # (999611, 5)
    # trans score
    data_df = data_df[data_df.label >= trans_score] # (999611, 5) trans_score=1, 所有评分都>=1
    # sort
    data_df = data_df.sort_values(by=['user_id', 'timestamp'])
    print('========negative sampling========')
    train_data, val_data, test_data = defaultdict(list), defaultdict(list), defaultdict(list)
    item_id_max = data_df['item_id'].max()  # item_id_max: 3952
    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()   # df: user_id对应的userid, itemid数据, pos_list: df里item_id列表
        def gen_neg():
            neg = pos_list[0]
            while neg in set(pos_list):
                neg = random.randint(1, item_id_max)
                return neg

        neg_list = [gen_neg() for i in range(len(pos_list)+test_neg_num)]
        for i in range(1, len(pos_list)):
            # hist_i = pos_list[:i]   # history_i: 取pos_list里i之前的所有元素
            # 对于每一个用户，取倒数第一个作为测试集，取倒数第二个作为验证集，其他数据作为训练集
            if i == len(pos_list)-1:
                test_data['user_id'].append(user_id)
                test_data['pos_id'].append(pos_list[i])
                test_data['neg_id'].append(neg_list[i:])    # neg_id有101个数
            elif i == len(pos_list)-2:
                val_data['user_id'].append(user_id)
                val_data['pos_id'].append(pos_list[i])
                val_data['neg_id'].append(neg_list[i])
            else:
                train_data['user_id'].append(user_id)
                train_data['pos_id'].append(pos_list[i])
                train_data['neg_id'].append(neg_list[i])
    # train_data: 981491条, test_data: 6040, val_data: 6040
    # feature columns
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    item_feat_col = [sparseFeature('user_id', user_num, embed_dim),
                     sparseFeature('item_id', item_num, embed_dim)]
    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    train = [np.array(train_data['user_id']), np.array(train_data['pos_id']),
             np.array(train_data['neg_id'])]
    val = [np.array(val_data['user_id']), np.array(val_data['pos_id']),
             np.array(val_data['neg_id'])]
    test = [np.array(test_data['user_id']), np.array(test_data['pos_id']),
             np.array(test_data['neg_id'])]
    print('========data preprocess end========')
    return item_feat_col, train, val, test


