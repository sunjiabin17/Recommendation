# Factorization Machines
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2


class FM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        '''

        :param feature_columns: a list containing dense and sparse column feature info
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        '''
        super(FM_Layer, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        # feature_length = 1296722 (1296709+13)
        self.feature_num = sum([feat['feat_num'] for feat in self.sparse_feature_columns]) \
                           + len(self.dense_feature_columns)
        self.k = k  # k = 10
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_num, 1),  # (1296722, 1)
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.k, self.feature_num),  # (10, 1296722) k = 10
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        '''
        :param inputs: [dense_feats, sparse_feats] shape=[(batchsize,13), (batchsize, 26)], batchsize=4096
        '''
        # dense_inputs.shape=(batchsize, 13), sparse_inputs.shape=(batchsize, 26)
        dense_inputs, sparse_inputs = inputs
        # one-hot encoding
        sparse_inputs = tf.concat(
            [tf.one_hot(sparse_inputs[:, i],
                        depth=self.sparse_feature_columns[i]['feat_num'])
             for i in range(sparse_inputs.shape[1])], axis=1    # sparse_inputs.shape[1] = 26
        )   # sparse_inputs.shape=(batchsize, 1296709) one-hot之后
        stack = tf.concat([dense_inputs, sparse_inputs], axis=1)    # stack.shape=(batchsize, 1296722) 1296709+13
        # first order
        first_order = self.w0 + tf.matmul(stack, self.w)    # first_order.shape=(batchsize, 1) --(batchsize, 1296722) multiply (1296722, 1)
        # second order
        second_order = 0.5 * tf.reduce_sum(
            tf.pow(tf.matmul(stack, tf.transpose(self.V)), 2) -     # shape=(batchsize, 10), (batchsize, 1296722) multiply (10, 1296722).T
            tf.matmul(tf.pow(stack, 2), tf.pow(tf.transpose(self.V), 2)),   # shape=(batchsize, 10), (batchsize, 1296722) multiply (10, 1296722).T
            axis=1, keepdims=True   # reduce_sum后的shape=(batchsize, 1)
        )
        outputs = first_order + second_order    # outputs.shape=(batchsize, 1)
        return outputs


class FM(tf.keras.Model):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        '''
        Factorization Machines
        :param feature_columns: a list containing dense and sparse column feature info
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        '''
        super(FM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.fm = FM_Layer(feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        '''
        :param inputs: [dense_feats, sparse_feats] shape=[(batchsize,13), (batchsize, 26)]
        '''
        fm_outputs = self.fm(inputs)    # fm_outputs.shape=(batchsize, 1)
        output = tf.nn.sigmoid(fm_outputs)
        return output   # output.shape=(batchsize, 1)

    def summary(self, **kwargs):
        dense_inputs = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()


import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from utils import create_criteo_dataset
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
    file = '../criteo_small.txt'
    read_part = True
    sample_num = 1000000
    test_size = 0.2

    k = 10

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10
    # ========================== Create dataset =======================
    # feature_columns:[[13个denseFeature], [26个sparseFeatures]]
    feature_columns, train, test = create_criteo_dataset(file=file,
                                                         read_part=read_part,
                                                         sample_num=sample_num,
                                                         test_size=test_size)
    train_X, train_y = train    # train_X:[dense_feats, sparse_feats] train_X.shape=[(800000,13), (800000, 26)], train_y.shape=(800000,)
    test_X, test_y = test       # test_X:[dense_feats, sparse_feats] train_X.shape=[(200000,13), (200000, 26)], test_y.shape=(200000,)

    # ============================Build Model==========================
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        # feature_columns:[[13个denseFeature], [26个sparseFeatures]]
        model = FM(feature_columns=feature_columns, k=k)
        model.summary()
        # ============================Compile============================
        model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                      metrics=[AUC()])
    # ============================model checkpoint======================
    # check_path = '../save/fm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ==============================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])
