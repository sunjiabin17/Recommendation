import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.regularizers import l2

class FFM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        '''

        :param feature_columns:
        :param k: latent vector
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        '''
        super(FFM_Layer, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.feature_num = sum([feat['feat_num'] for feat in self.sparse_feature_columns]) \
                            + len(self.dense_feature_columns)
        self.field_num = len(self.dense_feature_columns) + len(self.sparse_feature_columns)

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_num, 1), # (1296722, 1)
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.v = self.add_weight(name='v', shape=(self.feature_num, self.field_num, self.k),    # (1296722, field_num, k)
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs    # shape = [batchsize, 13], [batchsize, 26]
        # one-hot encoding
        sparse_inputs = tf.concat(
            [tf.one_hot(sparse_inputs[:, i],
                        depth=self.sparse_feature_columns[i]['feat_num'])
             for i in range(sparse_inputs.shape[1])], axis=1  # sparse_inputs.shape[1] = 26
        )   # sparse_inputs.shape=(batchsize, 1296709) one-hot之后
        stack = tf.concat([dense_inputs, sparse_inputs], axis=1)    # # stack.shape=(batchsize, 1296722) 1296709+13
        first_order = self.w0 + tf.matmul(tf.concat(stack, axis=-1), self.w) # shape=(batchsize, 1)

        second_order = 0
        field_f = tf.tensordot(stack, self.v, axes=[1, 0]) # (batchsize, 1296722) tensordot (1296722, field_num, k) = [batchsize, field_num, k]
        for i in range(self.field_num):
            for j in range(i+1, self.field_num):
                second_order += tf.reduce_sum(  # field_f[:, i].shape=[batchsize, k]
                    tf.multiply(field_f[:, i], field_f[:, j]), axis=1, keepdims=True # [batchsize, k]
                )# [batchsize, 1]
        return first_order + second_order


class FFM(tf.keras.Model):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        '''
        FFM architecture
        :param feature_columns: a list containing dense and sparse column feature info
        :param k: latent vector
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        '''
        super(FFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.ffm = FFM_Layer(feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):   # inputs.shape=([batchsize, 13], [batchsize, 26])
        result_ffm = self.ffm(inputs)
        outputs = tf.nn.sigmoid(result_ffm) # (batchsize, 1)
        return outputs

    def summary(self, **kwargs):
        dense_inputs = Input(shape=(len(self.dense_feature_columns), ), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns), ), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs],
                       outputs=self.call([dense_inputs, sparse_inputs])).summary()


from utils import create_criteo_dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.losses import binary_crossentropy
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    file = '../criteo_small.txt'
    read_part = True
    sample_num = 100000
    test_size = 0.2

    k = 8

    learning_rate = 0.001
    batch_size = 512
    epochs = 5

    # create dataset
    feature_columns, train, test = create_criteo_dataset(file=file,
                                                         read_part=read_part,
                                                         sample_num=sample_num,
                                                         test_size=test_size)
    train_X, train_y = train    #train_X.shape=[(80000,13),(80000,26)], train_y.shape=(80000,)
    test_X, test_y = test

    model = FFM(feature_columns, k)
    model.summary()

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=binary_crossentropy,
                  metrics=[AUC()])

    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )

    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])