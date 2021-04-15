# 矩阵分解推荐---隐语义模型
import random
import pickle
import pandas as pd
import numpy as np
import math


class LFM:
    def __init__(self):
        self.class_count = 5    # 隐语义模型中隐类别的数量
        self.iter_count = 5     # 迭代次数
        self.lr = 0.02          # 学习率
        self.lam = 0.01         # lambda 正则项系数
        self._init_model()
        self._init_Params()

    def _init_model(self):
        file_path = 'ml-1m/ratings.dat'
        self.frame = pd.read_csv(file_path, sep='::', engine='python')
        self.frame.columns = ['UserId', 'ItemId', 'Rating', 'TimeStamp']
        self.user_ids = set(self.frame['UserId'].values)
        self.item_ids = set(self.frame['ItemId'].values)
        self.items_dict = {user_id: self.get_pos_neg_item(user_id) for user_id in list(self.user_ids)}

    def _init_Params(self):
        array_p = np.random.randn(len(self.user_ids), self.class_count)
        array_q = np.random.randn(len(self.item_ids), self.class_count)
        self.p = pd.DataFrame(array_p, columns=range(0, self.class_count), index=list(self.user_ids))
        self.q = pd.DataFrame(array_q, columns=range(0, self.class_count), index=list(self.item_ids))

    def get_pos_neg_item(self, user_id):
        '''
        pos_item: user对该item评分过
        neg_item: user没有对该item评分, (没看过该电影)
        '''
        pos_item_ids = set(self.frame[self.frame['UserId'] == user_id]['ItemId'])
        neg_item_ids = set(self.frame['ItemId']) - pos_item_ids
        neg_item_ids = list(neg_item_ids)[:len(pos_item_ids)]

        item_dict = {}
        for item in pos_item_ids: item_dict[item] = 1
        for item in neg_item_ids: item_dict[item] = 0
        return item_dict

    def _predict(self, user_id, item_id):
        # 计算user和item之间的兴趣度
        p = np.mat(self.p.loc[user_id].values)
        q = np.mat(self.q.loc[item_id].values).T
        r = (p * q).sum()
        logit = 1.0 / (1 + math.exp(-r))
        return logit

    def loss(self, user_id, item_id, y, step):
        # MSE Loss
        err = y - self._predict(user_id, item_id)
        print('Step={}, user_id={},item_id={}, y={}, loss={}'.\
              format(step, user_id, item_id, y, err))
        return err

    def optimize(self, user_id, item_id, err):
        gradient_p = -err * self.q.loc[item_id].values
        l2_p = self.lam * self.p.loc[user_id].values
        delta_p = self.lr * (gradient_p + l2_p)

        gradient_q = -err * self.p.loc[user_id].values
        l2_q = self.lam * self.q.loc[item_id].values
        delta_q = self.lr * (gradient_q + l2_q)

        self.p.loc[user_id] -= delta_p
        self.q.loc[item_id] -= delta_q

    def train(self):
        for step in range(0, self.iter_count):
            for user_id, items_dict in self.items_dict.items():
                item_ids = list(items_dict.keys())
                random.shuffle(item_ids)
                for item_id in item_ids:
                    err = self.loss(user_id, item_id, items_dict[item_id], step)
                    self.optimize(user_id, item_id, err)
            self.lr *= 0.9
        self.save()

    def predict(self, user_id, top_n=10):
        self.load()
        user_item_ids = set(self.frame[self.frame['UserId'] == user_id]['ItemId'])
        other_item_ids = set(self.frame['UserId']) - user_item_ids
        interest_list = [self._predict(user_id, item_id) for item_id in other_item_ids]
        candidates = sorted(zip(list(other_item_ids), interest_list), key=lambda x: x[1], reverse=True)
        return candidates[:top_n]

    def save(self):
        f = open('lfm.model', 'wb')
        pickle.dump((self.p, self.q), f)
        f.close()

    def load(self):
        f = open("lfm.model", 'r')
        self.p, self.q = pickle.load(f)
        f.close()

if __name__ == '__main__':
    print('init')
    lfm = LFM()
    print('train')
    lfm.train()
    print('predict')
    movies = lfm.predict(user_id=1)
    for movie in movies:
        print(movie)