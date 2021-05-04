# Neural network-based Collaborative Filtering
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Input
from tensorflow.keras.layers import Layer, Dropout
from utils import create_ml1m_dataset

class DNN(Layer):
    '''
    Deep part
    '''
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0., **kwargs):
        '''
        DNN part
        :param hidden_units: a list. List of hidden layer units' number
        :param activation: a string. Activation function
        :param dnn_dropout: a scalar. dropout number
        :param kwargs:
        '''
        super(DNN, self).__init__(**kwargs)
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x


class NCF(Model):
    def __init__(self, feature_columns, hidden_units=None, dropout=0.2, activation='relu', embed_reg=1e-6, **kwargs):
        '''
        NCF Model
        :param feature_columns: a list. user feature columns + item feature columns
        :param hidden_units: a list. List of hidden layer units' number
        :param dropout: a scalar.
        :param activation: a string.
        :param embed_reg: a scalar. The regularizer of embedding
        :param kwargs:
        '''
        super(NCF, self).__init__(**kwargs)
        if hidden_units is None:
            hidden_units = [64, 32, 16, 8]

        # feature columns
        self.user_fea_col, self.item_fea_col = feature_columns
        # MF user embedding
        self.mf_user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                           input_length=1,
                                           output_dim=self.user_fea_col['embed_dim'],
                                           embeddings_initializer='random_normal',
                                           embeddings_regularizer=l2(embed_reg))
        # MF item embedding
        self.mf_item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                           input_length=1,
                                           output_dim=self.item_fea_col['embed_dim'],
                                           embeddings_initializer='random_normal',
                                           embeddings_regularizer=l2(embed_reg))
        # MLP user embedding
        self.mlp_user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                            input_length=1,
                                            output_dim=self.user_fea_col['embed_dim'],
                                            embeddings_initializer='random_normal',
                                            embeddings_regularizer=l2(embed_reg))
        # MLP item embedding
        self.mlp_item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                            input_length=1,
                                            output_dim=self.item_fea_col['embed_dim'],
                                            embeddings_initializer='random_normal',
                                            embeddings_regularizer=l2(embed_reg))
        # dnn
        self.dnn = DNN(hidden_units, activation=activation, dnn_dropout=dropout)
        self.dense = Dense(1, activation=None)

    def call(self, inputs):
        # inputs: [user_id, pos_id, neg_id] x batch_size
        user_inputs, pos_inputs, neg_inputs = inputs    # (None, 1), (None, 1), (None, 1/101)
        # user info
        mf_user_embed = self.mf_user_embedding(user_inputs)
        mlp_user_embed = self.mlp_user_embedding(user_inputs)
        # item info
        mf_pos_embed = self.mf_item_embedding(pos_inputs)
        mf_neg_embed = self.mf_item_embedding(neg_inputs)
        mlp_pos_embed = self.mlp_item_embedding(pos_inputs)
        mlp_neg_embed = self.mlp_item_embedding(neg_inputs)
        # MF
        mf_pos_vector = tf.nn.sigmoid(tf.multiply(mf_user_embed, mf_pos_embed))
        mf_neg_vector = tf.nn.sigmoid(tf.multiply(mf_user_embed, mf_neg_embed))
        # MLP
        mlp_pos_vector = tf.concat([mlp_user_embed, mlp_pos_embed], axis=-1)
        mlp_neg_vector = tf.concat([tf.tile(mlp_user_embed, multiples=[1, mlp_neg_embed.shape[1], 1]),
                                    mlp_neg_embed], axis=-1)
        mlp_pos_vector = self.dnn(mlp_pos_vector)
        mlp_neg_vector = self.dnn(mlp_neg_vector)
        # concat
        pos_vector = tf.concat([mf_pos_vector, mlp_pos_vector], axis=-1)
        neg_vector = tf.concat([mf_neg_vector, mlp_neg_vector], axis=-1)
        # result
        pos_logits = tf.squeeze(self.dense(pos_vector), axis=-1)
        neg_logits = tf.squeeze(self.dense(neg_vector), axis=-1)
        # loss
        losses = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_logits)) -
                                tf.math.log(1-tf.nn.sigmoid(neg_logits))) / 2
        self.add_loss(losses)
        logits = tf.concat([pos_logits, neg_logits], axis=-1)
        return logits

    def summary(self):
        user_inputs = Input(shape=(1,), dtype=tf.int32)
        pos_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_inputs = Input(shape=(1,), dtype=tf.int32)
        Model(inputs=[user_inputs, pos_inputs, neg_inputs],
              outputs=self.call([user_inputs, pos_inputs, neg_inputs])).summary()


# def test_model():
#     user_features = {'feat': 'user_id', 'feat_num': 100, 'embed_dim': 8}
#     item_features = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}
#     features = [user_features, item_features]
#     model = NCF(features)
#     model.summary()
#
# test_model()

def evaluate_model(model, test, K):
    '''
    evaluate model
    :param model:
    :param test:
    :param K: top K
    :return: hit rate, ndcg
    '''
    pred_y = -model.predict(test)
    rank = pred_y.argsort().argsort()[:, 0]
    hr, ndcg = 0.0, 0.0
    for r in rank:
        if r < K:
            hr += 1
            ndcg += 1 / np.log2(r + 2)
    return hr / len(rank), ndcg / len(rank)


import os
from time import time
from tensorflow.keras.optimizers import Adam

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    file = '../data/ml-1m/ratings.dat'
    trans_score = 1
    test_neg_num = 100

    embed_dim = 8
    hidden_units = [256, 128, 64]
    embed_reg = 1e-6
    activation = 'relu'
    dropout = 0.2
    K = 10

    learning_rate = 0.001
    epochs = 20
    batch_size = 32

    # create dataset
    feature_columns, train, val, test = create_ml1m_dataset(file, trans_score, embed_dim, test_neg_num)
    # feature_columns:feat, feat_num, embed_dim
    # train:[user_id, pos_id, neg_id]x981491
    # val: [user_id, pos_id, neg_id]x6040
    # test: [user_id, pos_id, neg_id(101个)]x6040
    # build model
    # mirrired_strategy = tf.distribute.MirroredStrategy()
    # with mirrired_strategy.scope():

    model = NCF(feature_columns, hidden_units, dropout, activation, embed_reg)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=learning_rate))

    results = []
    # fit
    for epoch in range(1, epochs + 1):
        t1 = time()
        model.fit(
            train,
            None,
            validation_data=(val, None),
            epochs=1,
            batch_size=batch_size
        )

        # test
        # t2 = time()
        # if epoch % 1 == 0:
        #     hit_rate, ndcg = evaluate_model(model, test, K)
        #     print('Iteration %d [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f'
        #           % (epoch, t2-t1, time()-t2, hit_rate, ndcg))
        #     results.append([epoch, t2-t1, time()-t2, hit_rate, ndcg])

    # write log
    pd.DataFrame(results, columns=['iteration', 'fit_time', 'evaluate_time', 'hit_rate', 'ndcg']) \
        .to_csv('log/NCF_log_dim_{}_K_{}.csv'.format(embed_dim, K), index=False)