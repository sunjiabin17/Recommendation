import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, BatchNormalization, Dense

from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input, PReLU, Dropout


class Attention_Layer(Layer):
    def  __init__(self, att_hidden_units, activation='sigmoid'):
        super(Attention_Layer, self).__init__()
        self.att_dense = [Dense(unit, activation=activation) for unit in att_hidden_units]
        self.att_final_dense = Dense(1)


    def call(self, inputs, **kwargs):
        # query: candidate item (None, d*2), d is the dim of embedding
        # key: hist items (None, seq_len, d*2)
        # value: hist items (None, seq_len, d*2)
        # mask: (None, seq_len)
        q, k, v, mask = inputs
        q = tf.tile(q, multiples=[1, k.shape[1]])   # (None, seq_len * d * 2)
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])   # (None, seq_len, d * 2)

        # q, k, out product should concat
        info = tf.concat([q, k, q-k, q*k], axis=-1) # (None, seq_len, d * 2 * 4) 4个concat

        # dense
        for dense in self.att_dense:
            info = dense(info)
        # info.shape=(None, seq_len, 40)
        outputs = self.att_final_dense(info)  # (None, seq_len, 1)
        outputs = tf.squeeze(outputs, axis=-1)  # (None, seq_len)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # (None, seq_len)
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)  # (None, seq_len) mask的值如果是0，说明是padding，取paddings的值，mask的值如果是1，说明是真值，取outputs的值

        # softmax
        outputs = tf.nn.softmax(logits=outputs)
        outputs = tf.expand_dims(outputs, axis=1)
        outputs = tf.matmul(outputs, v) # (None, 1, seq_len) matmul (None, seq_len, d*2) = (None, 1, d*2)
        outputs = tf.squeeze(outputs, axis=1)   # (None, d*2)

        return outputs


class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False, epsilon=1e-8)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, s, **kwargs):
        s_normed = self.bn(s)   # 每个batch转换为均值为0方差为1
        ps = tf.sigmoid(s_normed)

        return self.alpha * (1.0 - ps) * s + ps * s


class DIN(Model):
    def __init__(self, feature_columns, behavior_feature_list, att_hidden_units=(80, 40),
                 ffn_hidden_units=(80, 40), att_activation='prelu', ffn_activation='prelu',
                 maxlen=40, dnn_dropout=0., embed_reg=1e-4):
        '''
        DIN
        :param feature_columns: a list. dense_feature_columns + sparse_feature_columns
        :param behavior_feature_list: a list. list of behavior feature names.
        :param att_hidden_units: a tuple or list. attention hidden units.
        :param ffn_hidden_units: a tuple or list. hidden units list of FFN.
        :param att_activation: a string. activation of attention.
        :param ffn_activation: a string. prelu or dice.
        :param maxlen: a scalar. maximum sequence length.
        :param dnn_dropout: a scalar. number of dropout.
        :param embed_reg: a scalar. regularizer of embedding.
        '''
        super(DIN, self).__init__()
        self.maxlen = maxlen    # 20
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns

        # len   behavior_feature_list: item_id
        self.other_sparse_len = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.dense_len = len(self.dense_feature_columns)
        self.behavior_len = len(behavior_feature_list)

        # other embedding layers
        self.embed_sparse_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embed_reg))
                                    for feat in self.sparse_feature_columns
                                    if feat['feat'] not in behavior_feature_list]

        # behavior embedding layers, item id and category id
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                           input_length=1,
                                           output_dim=feat['embed_dim'],
                                           embeddings_initializer='random_uniform',
                                           embeddings_regularizer=l2(embed_reg))
                                 for feat in self.sparse_feature_columns
                                 if feat['feat'] in behavior_feature_list]

        # attention layer
        self.attention_layer = Attention_Layer(att_hidden_units, att_activation)
        self.bn = BatchNormalization(trainable=True)

        # ffn
        self.ffn = [Dense(unit, activation=PReLU() if ffn_activation == 'prelu' else Dice())
                    for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(1)

    # train_X: [[0,...], [0,...], [hist], [target_item]]
    def call(self, inputs, training=None, mask=None):
        # dense_inputs and sparse_inputs is empty
        # seq_inputs seq_inputs: (None, maxlen, 2) maxlen是浏览过的历史物品padding后的结果， 2是物品id和种类id，相当于DIN论文中的历史行为
        # item_inputs: 要预测的当前物品，相当于DIN论文中的广告
        dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs

        # attention --> mask, if the element of seq_inputs is equal 0, it must be filled in.
        mask = tf.cast(tf.not_equal(seq_inputs[:, :, 0], 0), dtype=tf.float32)  # (None, maxlen) 标记是否为padding 0为padding，1为原数据值
        # other
        other_info = dense_inputs
        for i in range(self.other_sparse_len):
            other_info = tf.concat([other_info, self.embed_sparse_layers[i](sparse_inputs[:, i])], axis=-1)

        # seq, item embedding and category embedding should concatenate
        seq_embed = tf.concat([self.embed_seq_layers[i](seq_inputs[:, :, i]) for i in range(self.behavior_len)], axis=-1)   # (None, maxlen, embed_dim * behavior_len=16)
        item_embed = tf.concat([self.embed_seq_layers[i](item_inputs[:, i]) for i in range(self.behavior_len)], axis=-1)    # (None, embed_dim * behavior_len=16)

        # att
        user_info = self.attention_layer([item_embed, seq_embed, seq_embed, mask])  # (None, d*2)

        # concat user_info(att hist), candidate item embedding, other features
        if self.dense_len > 0 or self.other_sparse_len > 0:
            info_all = tf.concat([user_info, item_embed, other_info], axis=-1)
        else:
            info_all = tf.concat([user_info, item_embed], axis=-1)
        info_all = self.bn(info_all)    # (None, 32)

        # ffn
        for dense in self.ffn:
            info_all = dense(info_all)
        # info_all.shape=(None, 64)
        info_all = self.dropout(info_all)
        outputs = tf.nn.sigmoid(self.dense_final(info_all)) # (None, 1)
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(self.dense_len, ), dtype=tf.float32)
        sparse_inputs = Input(shape=(self.other_sparse_len, ), dtype=tf.int32)
        seq_inputs = Input(shape=(self.maxlen, self.behavior_len), dtype=tf.int32)
        item_inputs = Input(shape=(self.behavior_len, ), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs, seq_inputs, item_inputs],
                       outputs=self.call([dense_inputs, sparse_inputs, seq_inputs, item_inputs])).summary()

def test_model():
    dense_features = [{'feat': 'a'}, {'feat': 'b'}]
    sparse_features = [{'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'cate_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'adv_id', 'feat_num': 100, 'embed_dim': 8}]
    behavior_list = ['item_id', 'cate_id']
    features = [dense_features, sparse_features]
    model = DIN(features, behavior_list, att_activation='sigmoid')
    model.summary()


# test_model()
