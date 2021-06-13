import tensorflow as tf
from time import time
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

from DIN import DIN
from utils import *

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    file = '../data/remap.pkl'
    maxlen = 20

    embed_dim = 8
    att_hidden_units = [80, 40]
    ffn_hidden_units = [256, 128, 64]
    dnn_dropout = 0.5
    att_activation = 'sigmoid'
    ffn_activation = 'prelu'

    learning_rate = 0.001
    batch_size = 1024
    epochs = 5

    feature_columns, behavior_list, train, val, test = create_amazon_electronic_dataset(file, embed_dim, maxlen)
    # train_X: [[0,...], [0,...], [history], [target_item]]
    # hist: 浏览过的历史商品
    # target_item: 目标商品
    # train_y: [label] 标记目标商品是否购买
    train_X, train_y = train
    val_X, val_y = val
    test_X, test_y = test

    model = DIN(feature_columns, behavior_list, att_hidden_units, ffn_hidden_units, att_activation,
                ffn_activation, maxlen, dnn_dropout)
    model.summary()

    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        validation_data=(val_X, val_y),
        batch_size=batch_size
    )
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])

