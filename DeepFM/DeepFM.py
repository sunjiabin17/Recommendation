import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Dense, Input, Layer
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from utils import create_criteo_dataset


class FM(Layer):
    '''
    wide part
    '''

    def __init__(self, k=10, w_reg=1e-4, v_reg=1e-4):
        '''
        Factorization Machine
        :param k: a scalar. the dimension of the latent vector.
        :param w_reg: a scalar. the regularization coefficient of parameter w.
        :param v_reg: a scalar. the regularization coefficient of parameter v.
        '''
        super(FM, self).__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer='random_uniform',
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.k, input_shape[-1]),
                                 initializer='random_uniform',
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        # first order
        first_order = self.w0 + tf.matmul(inputs, self.w)
        # second order
        second_order = 0.5 * tf.reduce_sum(
            tf.pow(tf.matmul(inputs, tf.transpose(self.V)), 2) -
            tf.matmul(tf.pow(inputs, 2), tf.pow(tf.transpose(self.V), 2)), axis=1, keepdims=True
        )
        return first_order + second_order


class DNN(Layer):
    '''
    Deep part
    '''

    def __init__(self, hidden_units, activation='relu', dnn_dropout=0.):
        '''
        Dnn part
        :param hidden_units: a list. List of hidden layer units' numbers.
        :param activation: a string. Activation function.
        :param dnn_dropout: a scalar. Dropout number
        '''
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x


class DeepFM(Model):
    def __init__(self, feature_columns, k=10, hidden_units=(200, 200, 200), dnn_dropout=0.,
                 activation='relu', fm_w_reg=1e-4, fm_v_reg=1e-4, embed_reg=1e-4):
        '''
        DeepFM
        :param feature_columns: a list. a list containing dense and sparse column feature information.
        :param k: a scalar. fm's latent vector number.
        :param hidden_units: a list. a list of dnn hidden units.
        :param dnn_dropout: a scalar. dropout of dnn.
        :param activation: a string. activation function of dnn.
        :param fm_w_reg: a scalar. regularizer of w in fm.
        :param fm_v_reg: a scalar. regularizer of v in fm.
        :param embed_reg: a scalar. regularizer of embedding.
        '''
        super(DeepFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.fm = FM(k, fm_w_reg, fm_v_reg)
        self.dnn = DNN(hidden_units, activation, dnn_dropout)
        self.dense = Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs
        sparse_embed = tf.concat([
            self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
            for i in range(sparse_inputs.shape[1])
        ], axis=-1)
        stack = tf.concat([dense_inputs, sparse_embed], axis=-1)
        # wide
        wide_outputs = self.fm(stack)
        # deep
        deep_outputs = self.dnn(stack)
        deep_outputs = self.dense(deep_outputs)

        outputs = tf.nn.sigmoid(tf.add(wide_outputs, deep_outputs))
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),),dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),),dtype=tf.int32)
        Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()


if __name__ == '__main__':
    file = '../data/criteo_small.txt'
    read_part = True
    sample_num = 100000
    test_size = 0.2

    embed_dim = 8
    k = 10
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 1024
    epochs = 10

    # create dataset
    feature_columns, train, test = create_criteo_dataset(file=file,
                                                         embed_dim=embed_dim,
                                                         read_part=read_part,
                                                         sample_num=sample_num,
                                                         test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test
    # build model
    model = DeepFM(feature_columns, k, hidden_units, dnn_dropout)
    model.summary()
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])

    # fit
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        # callbacks=[EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)],
        batch_size=batch_size,
        validation_split=0.1
    )
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])
