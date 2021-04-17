import pandas as pd
from tqdm import tqdm


def sparseFeature(feat, feat_num, embed_dim=4):
    '''
    创建稀疏矩阵的字典
    :param feat: feature name
    :param feat_num: total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    '''
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    '''
    创建稠密矩阵的字典
    :param feat: feature name
    :return:
    '''
    return {'feat': feat}


def create_explicit_dataset(file, latent_dim=4, test_size=0.2):
    '''
    把每个用户按时间戳排序的最后20%作为测试集
    :param file: dataset path
    :param latent_dim: latent factor
    :param test_size: ratio of test dataset
    :return: user_num, item_num, train_df, test_df
    '''
    data_df = pd.read_csv(file, sep='::', engine='python',
                          names=['UserId', 'MovieId', 'Rating', 'Timestamp'])
    data_df['avg_score'] = data_df.groupby(by='UserId')['Rating'].transform('mean')
    # feature columns
    user_num, item_num = data_df['UserId'].max() + 1, data_df['MovieId'].max() + 1
    feature_columns = [[denseFeature('avg_score')],
                       [sparseFeature('user_id', user_num, latent_dim),
                       sparseFeature('item_id', item_num, latent_dim)]]
    #split train and test
    watch_count = data_df.groupby(by='UserId')['MovieId'].agg('count')
    test_df = pd.concat([
        data_df[data_df.UserId == i].iloc[int(0.8*watch_count[i]):] for i in tqdm(watch_count.index)], axis=0)
    test_df = test_df.reset_index()
    train_df = data_df.drop(labels=test_df['index'])
    train_df = train_df.drop(['Timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)
    test_df = test_df.drop(['Timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)

    train_X = [train_df['avg_score'].values, train_df[['UserId', 'MovieId']].values]
    train_y = train_df['Rating'].values.astype('int32')
    test_X = [test_df['avg_score'].values, test_df[['UserId', 'MovieId']].values]
    test_y = test_df['Rating'].values.astype('int32')
    return feature_columns, (train_X, train_y), (test_X, test_y)

