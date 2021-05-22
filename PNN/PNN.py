import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Layer, Dropout

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from utils import create_criteo_dataset


class DNN(Layer):
    '''
    Deep Neural Network
    '''
    def __init__(self, hidden_units, activation='relu', dropout=0.):
        '''

        :param hidden_units: a list. neural network hidden units.
        :param activation: a string. activation function of dnn.
        :param dropout: a scalar. dropout number
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


class PNN(keras.Model):
    def __init__(self, feature_columns, hidden_units, mode='in', dnn_dropout=0.,
                 activation='relu', embed_reg=1e-4, w_z_reg=1e-4, w_p_reg=1e-4, l_b_reg=1e-4):
        '''
        Product_based Neural Network
        :param feature_columns: a list. dense_feature_columns + sparse_feature_columns
        :param hidden_units: a list. neural network hidden units.
        :param mode: a string. 'in' IPNN or 'out' OPNN.
        :param dnn_dropout: a scalar. dropout of dnn.
        :param activation: a string. activation function of dnn.
        :param embed_reg: a scalar. regularizer of embedding.
        :param w_z_reg: a scalar. regularizer of w_z in product layer.
        :param w_p_reg: a scalar. regularizer of w_p in product layer.
        :param l_b_reg: a scalar. regularizer of l_b in product layer.
        '''
        super(PNN, self).__init__()
        # inner product or outer product
        self.mode = mode
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        # the number of feature fields
        self.field_num = len(self.sparse_feature_columns)
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']
        # the embedding dimension of each feature field must be the same
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }

        # parameters
        self.w_z = self.add_weight(name='w_z',
                                   shape=(self.field_num, self.embed_dim, hidden_units[0]),
                                   initializer='random_uniform',
                                   regularizer=l2(w_z_reg),
                                   trainable=True)
        if mode == 'in':
            self.w_p = self.add_weight(name='w_p',
                                       shape=(self.field_num * (self.field_num - 1) // 2, self.embed_dim, hidden_units[0]),
                                       initializer='random_uniform',
                                       regularizer=l2(w_p_reg),
                                       trainable=True)
        else:
            self.w_p = self.add_weight(name='w_p',
                                       shape=(self.field_num * (self.field_num - 1) // 2, self.embed_dim, self.embed_dim, hidden_units[0]),
                                       initializer='random_uniform',
                                       regularizer=l2(w_p_reg),
                                       trainable=True)

        self.l_b = self.add_weight(name='l_b',
                                   shape=(hidden_units[0], ),
                                   initializer='random_uniform',
                                   regularizer=l2(l_b_reg),
                                   trainable=True)
        # dnn
        self.dnn_network = DNN(hidden_units[1:], activation, dnn_dropout)
        self.dense_final = Dense(1)

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs        # (None, 13)  (None, 26)
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])   # {list: 26} (None, 8)
                 for i in range(sparse_inputs.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2]) # (None, field_num, embed_dim) (None, 26, 8)
        # product layer
        row = []
        col = []
        for i in range(len(self.sparse_feature_columns) - 1):
            for j in range(i + 1, len(self.sparse_feature_columns)):
                row.append(i)   # {list: 325} [0,0,0,0,..., 1,1,1,1,..., 2,2,2,2,....]
                col.append(j)   # {list: 325} [1,2,3,4,..., 2,3,4,5,..., 3,4,5,6,....]
        p = tf.gather(embed, row, axis=1)   # (None, 325, 8)
        q = tf.gather(embed, col, axis=1)   # {None, 325, 8)
        if self.mode == 'in':
            l_p = tf.tensordot(p * q, self.w_p, axes=2) # (None, 325, 8) * (325, 8, 256) = (None, 256) a的最后两维和b的前两维相乘
        else:   # out
            u = tf.expand_dims(q, 2)
            v = tf.expand_dims(p, 2)
            l_p = tf.tensordot(tf.matmul(tf.transpose(u, [0, 1, 3, 2]), v), self.w_p, axes=3)

        l_z = tf.tensordot(embed, self.w_z, axes=2) # (None, 26, 8) * (26, 8, 256) = (None, 256)
        l_1 = tf.nn.relu(tf.concat([l_z + l_p + self.l_b, dense_inputs], axis=-1))  # (None, 256+13=269)
        # dnn_layer
        dnn_x = self.dnn_network(l_1)   # (None, 269) -> (None, 64)
        outputs = tf.nn.sigmoid(self.dense_final(dnn_x))    # (None, 1)
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns), ), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns), ), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs])).summary()


if __name__ == '__main__':
    file = '../data/criteo_small.txt'
    read_part = True
    sample_num = 500000
    test_size = 0.2

    embed_dim = 8
    mode = 'out'
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 1024
    epochs = 5

    # create dataset
    feature_columns, train, test = create_criteo_dataset(file=file,
                                                         embed_dim=embed_dim,
                                                         read_part=read_part,
                                                         sample_num=sample_num,
                                                         test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test

    # build model
    model = PNN(feature_columns=feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout)
    model.summary()
    # compile
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate), metrics=[AUC()])

    # fit
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_auc', patience=2, restore_best_weights=True)],
        batch_size=batch_size,
        validation_split=0.1
    )

    # test
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])