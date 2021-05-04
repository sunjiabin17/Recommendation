# Wide & Deep 进阶--- Deep & Cross

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Layer, Dropout

from utils import create_criteo_dataset

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class CrossNetwork(Layer):
    '''
    Cross Network
    '''
    def __init__(self, layer_num, reg_w=1e-4, reg_b=1e-4):
        '''

        :param layer_num: a scalar. the depth of cross network.
        :param reg_w: a scalar. the regularization of w.
        :param reg_b: a scalar. the regularization of b.
        '''
        super(CrossNetwork, self).__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.cross_weights = [
            self.add_weight(name='w_' + str(i),
                            shape=(dim, 1),
                            initializer='random_uniform',
                            regularizer=l2(self.reg_w),
                            trainable=True)
            for i in range(self.layer_num)
        ]
        self.cross_bias = [
            self.add_weight(name='b_' + str(i),
                            shape=(dim, 1),
                            initializer='random_uniform',
                            regularizer=l2(self.reg_b),
                            trainable=True)
            for i in range(self.layer_num)
        ]

    def call(self, inputs, **kwargs):
        x_0 = tf.expand_dims(inputs, axis=2)    #(None, dim, 1)
        x_l = x_0
        for i in range(self.layer_num):
            x_l1 = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0])    # (None, dim, 1) * (dim, 1) [1, 0]-> (None, 1, 1)
            x_l = tf.matmul(x_0, x_l1) + self.cross_bias[i] + x_l       # tf.matmul( (None, dim, 1), (None, 1, 1) ) -> (None, dim, 1)
        x_l = tf.squeeze(x_l, axis=2)
        return x_l  # (None, dim)


class DNN(Layer):
    '''
    Deep network
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


class DCN(keras.Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=1e-4, cross_w_reg=1e-4, cross_b_reg=1e-4):
        '''
        Deep&Cross Network
        :param feature_columns: a list. dense_feature_columns + sparse_feature_columns.
        :param hidden_units: a list. neural network hidden units.
        :param activation: a string. activation function of dnn.
        :param dnn_dropout: a scalar. dropout of dnn.
        :param embed_reg: a scalar. the regularizer of embedding.
        :param cross_w_reg: a scalar. the regularizer of cross network.
        :param cross_b_reg: a scalar. the regularizer of cross network.
        '''
        super(DCN, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.layer_num = len(hidden_units)
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.cross_network = CrossNetwork(self.layer_num, cross_w_reg, cross_b_reg)
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.dense_final = Dense(1)

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                   for i in range(sparse_inputs.shape[1])], axis=-1)
        x = tf.concat([sparse_embed, dense_inputs], axis=-1)
        # Cross Network
        cross_x = self.cross_network(x) # (None, 221)
        # DNN
        dnn_x = self.dnn_network(x)     # (None, 64)
        # Concatenate
        total_x = tf.concat([cross_x, dnn_x], axis=-1)
        outputs = tf.nn.sigmoid(self.dense_final(total_x))
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns), ), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns), ), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs])).summary()



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    file = '../data/criteo_small.txt'
    read_part = True
    sample_num = 1000000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 1024
    epochs = 5

    # create dataset
    feature_columns, train, test = create_criteo_dataset(file,
                                                         embed_dim,
                                                         read_part,
                                                         sample_num,
                                                         test_size)
    train_X, train_y = train
    test_X, test_y = test

    # Build Model
    model = DCN(feature_columns, hidden_units, dnn_dropout=dnn_dropout)
    model.summary()

    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])


    # fit
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
        batch_size=batch_size,
        validation_split=0.1
    )
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])






