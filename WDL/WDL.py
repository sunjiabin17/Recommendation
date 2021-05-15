import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Embedding, Concatenate, Dropout, Input, Layer
from tensorflow.keras.regularizers import l2
from utils import *

import os
import pandas as pd
from time import time
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.activations import sigmoid


class Linear(Layer):
    def __init__(self):
        super(Linear, self).__init__()
        self.dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        result = self.dense(inputs)
        return result


class DNN(Layer):
    '''
    Deep Neural Network
    '''
    def __init__(self, hidden_units, activation='relu', dropout=0.):
        '''

        :param hidden_units: a list. neural network hidden units.
        :param activation: a string. activation function of dnn.
        :param dropout: a scalar. dropout number.
        '''
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x


class WideDeep(tf.keras.Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=1e-4):
        '''
        Wide&Deep
        :param feature_columns: a list. dense_feature_columns + sparse_feature_columns.
        :param hidden_units: a list. neural network hidden units.
        :param activation: a string. activation function of dnn.
        :param dnn_dropout: a scalar. dropout of dnn.
        :param embed_reg: a scalar. the regularization of embedding.
        '''
        super(WideDeep, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                        input_length=1,
                                        output_dim=feat['embed_dim'],
                                        embeddings_initializer='random_uniform',
                                        embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.linear = Linear()
        self.final_dense = Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs    # [None, 13] {None, 26]
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])  # [None, 208]
                                  for i in range(sparse_inputs.shape[1])], axis=-1)
        x = tf.concat([sparse_embed, dense_inputs], axis=-1)   # [None, 221]

        # Wide
        wide_out = self.linear(dense_inputs)    # [None, 1]
        # Deep
        deep_out = self.dnn_network(x)  # [None, 64] (256->128->64)
        deep_out = self.final_dense(deep_out)   # [None, 1]
        # out
        outputs = sigmoid(0.5 * wide_out + 0.5 * deep_out)    # [None, 1]
        return outputs

    def summary(self, **kwargs):
        dense_inputs = Input(shape=(len(self.dense_feature_columns), ), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns), ), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs])).summary()



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
    file = '../data/criteo_small.txt'
    read_part = True
    sample_num = 500000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 1024
    epochs = 5

    starttime = time()
    # create dataset
    feature_columns, train, test = create_criteo_dataset(file, embed_dim, read_part, sample_num, test_size)

    train_X, train_y = train
    test_X, test_y = test

    # build model
    model = WideDeep(feature_columns=feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout)
    model.summary()

    # compile
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])

    # model checkpoint
    # check_path = './wide_and_deep_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)

    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)],
        batch_size=batch_size,
        validation_split=0.1
    )
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])
    print('time cost: ', time() - starttime)