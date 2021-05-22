import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Embedding, ReLU, Layer, Dropout

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from utils import create_criteo_dataset


class ResidualUnits(Layer):
    '''
    residual units
    '''
    def __init__(self, hidden_unit, dim_stack):
        '''

        :param hidden_unit: a list. neural network hidden units.
        :param dim_stack: a scalar. dimension of inputs units.
        '''
        super(ResidualUnits, self).__init__()
        self.layer1 = Dense(units=hidden_unit, activation='relu')
        self.layer2 = Dense(units=dim_stack, activation=None)
        self.relu = ReLU()

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.layer1(x)
        x = self.layer2(x)
        outputs = self.relu(x + inputs)
        return outputs


class DeepCrossing(keras.Model):
    def __init__(self, feature_columns, hidden_units, res_dropout=0., embed_reg=1e-4):
        '''
        Deep&Crossing
        :param feature_columns: a list. dense_feature_columns + sparse_feature_columns.
        :param hidden_units: a list. neural network hidden units.
        :param res_dropout: a scalar. dropout of resnet.
        :param embed_reg: a scalar. regularizer of embedding.
        '''
        super(DeepCrossing, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        # the total length of embedding layers
        embed_dim = sum([feat['embed_dim'] for feat in self.sparse_feature_columns])
        # the dimension of stack layers
        dim_stack = len(self.dense_feature_columns) + embed_dim
        self.res_network = [ResidualUnits(unit, dim_stack) for unit in hidden_units]
        self.res_dropout = Dropout(res_dropout)
        self.dense = Dense(1)

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs
        sparse_embed = tf.concat([
            self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
            for i in range(sparse_inputs.shape[1])
        ], axis=-1)
        stack = tf.concat([sparse_embed, dense_inputs], axis=-1)
        r = stack
        for res in self.res_network:
            r = res(r)
        r = self.res_dropout(r)
        outputs = tf.nn.sigmoid(self.dense(r))
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs])).summary()


if __name__ == '__main__':
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

    # create dataset
    feature_columns, train, test = create_criteo_dataset(file=file,
                                                         embed_dim=embed_dim,
                                                         read_part=read_part,
                                                         sample_num=sample_num,
                                                         test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test

    # build model
    model = DeepCrossing(feature_columns, hidden_units)
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