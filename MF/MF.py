# 矩阵分解
# Matrix Factorization Techniques for Recommender Systems
from utils import *
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import  binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 屏蔽通知信息和警告信息


class MF_layer(Layer):
    def __init__(self, user_num, item_num, latent_dim, use_bias=False, user_reg=1e-4, item_reg=1e-4,
                 user_bias_reg=1e-4, item_bias_reg=1e-4):
        '''

        :param user_num: user length
        :param item_num: item length
        :param latent_dim: latent number
        :param use_bias: using bias or not
        :param user_reg: regularization of user
        :param item_reg: regularization of item
        :param user_bias_reg: regularization of user bias
        :param item_bias_reg: regularization of item bias
        '''
        super(MF_layer, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.use_bias = use_bias
        self.user_reg = user_reg
        self.item_reg = item_reg
        self.user_bias_reg = user_bias_reg
        self.item_bias_reg = item_bias_reg

    def build(self, input_shape):
        self.p = self.add_weight(name='user_latent_matrix',
                                 shape=(self.user_num, self.latent_dim),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.user_reg),
                                 trainable=True)
        self.q = self.add_weight(name='item_latent_matrix',
                                 shape=(self.item_num, self.latent_dim),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.item_reg),
                                 trainable=True)
        self.user_bias = self.add_weight(name='user_bias',
                                 shape=(self.user_num, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.user_bias_reg),
                                 trainable=self.use_bias)
        self.item_bias = self.add_weight(name='item_bias',
                                 shape=(self.item_num, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.item_bias_reg),
                                 trainable=self.use_bias)

    def call(self, inputs, **kwargs):
        user_id, item_id, avg_score = inputs
        # MF
        latent_user = tf.nn.embedding_lookup(params=self.p, ids=user_id)
        latent_item = tf.nn.embedding_lookup(params=self.q, ids=item_id)
        outputs = tf.reduce_sum(tf.multiply(latent_user, latent_item), axis=1, keepdims=True)
        # MF-bias
        user_bias = tf.nn.embedding_lookup(params=self.user_bias, ids=user_id)
        item_bias = tf.nn.embedding_lookup(params=self.item_bias, ids=item_id)
        bias = tf.reshape((avg_score + user_bias + item_bias), shape=(-1, 1))

        outputs = bias + outputs if self.use_bias else outputs
        return outputs

    def summary(self):
        user_id = tf.keras.Input(shape=(), dtype=tf.int32)
        item_id = tf.keras.Input(shape=(), dtype=tf.int32)
        avg_score = tf.keras.Input(shape=(), dtype=tf.float32)
        tf.keras.Model(inputs=[user_id, item_id, avg_score], outputs=self.call([user_id, item_id, avg_score])).summary()


class MF(tf.keras.Model):
    def __init__(self, feature_columns, use_bias=False, user_reg=1e-4, item_reg=1e-4,
                 user_bias_reg=1e-4, item_bias_reg=1e-4):
        '''
        MF Model
        :param feature_columns: dense_feature_columns + sparse_feature_columns
        :param implicit: implicit or not
        :param use_bias: using bias or not
        :param user_reg: regularization of user
        :param item_reg: regularization of item
        :param user_bias_reg: regularization of user bias
        :param item_bias_reg: regularization of item bias
        '''
        super(MF, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        num_users, num_items = self.sparse_feature_columns[0]['feat_num'], \
                                self.sparse_feature_columns[1]['feat_num']
        latent_dim = self.sparse_feature_columns[0]['embed_dim']
        self.mf_layer = MF_layer(num_users, num_items, latent_dim, use_bias,
                                 user_reg, item_reg, user_bias_reg, item_bias_reg)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        user_id, item_id = sparse_inputs[:, 0], sparse_inputs[:, 1]
        avg_score = dense_inputs
        outputs = self.mf_layer([user_id, item_id, avg_score])
        return outputs

    def summary(self, **kwargs):
        dense_inputs = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()


if __name__ == '__main__':
    file = '../data/ml-1m/ratings.dat'
    test_size = 0.2
    latent_dim = 32

    use_bias = True

    learning_rate = 0.001
    batch_size = 512
    epochs = 10

    # create dataset
    feature_columns, train, test = create_explicit_dataset(file, latent_dim, test_size)
    train_X, train_y = train
    test_X, test_y = test

    model = MF(feature_columns, use_bias=use_bias)
    model.summary()

    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate),
                  metrics=['mse'])
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )
    print('test_rmse: %f' % np.sqrt(model.evaluate(test_X, test_y)[1]))