import tensorflow as tf
from layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, \
    optimized_trilinear_for_attention


class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True, demo=False, graph=None):
        self.config = config
        self.demo = demo
        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():

            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            if self.demo:
                self.c = tf.placeholder(tf.int32, [None, config.test_para_limit], "context")
                self.q = tf.placeholder(tf.int32, [None, config.test_ques_limit], "question")
                self.ch = tf.placeholder(tf.int32, [None, config.test_para_limit, config.char_limit], "context_char")
                self.qh = tf.placeholder(tf.int32, [None, config.test_ques_limit, config.char_limit], "question_char")
                self.y1 = tf.placeholder(tf.int32, [None, config.test_para_limit], "answer_index1")
                self.y2 = tf.placeholder(tf.int32, [None, config.test_para_limit], "answer_index2")
            else:
                self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()

            # self.word_unk = tf.get_variable("word_unk", shape = [config.glove_dim], initializer=initializer())
            # word embeddings and character embeddings
            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
                word_mat, dtype=tf.float32), trainable=False)
            self.char_mat = tf.get_variable(
                "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

            # context mask and question mask
            self.c_mask = tf.cast(self.c, tf.bool)
            self.q_mask = tf.cast(self.q, tf.bool)
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

            if opt:
                # num_batch and character limit
                N, CL = config.batch_size if not self.demo else 1, config.char_limit
                # 最大长度
                self.c_maxlen = tf.reduce_max(self.c_len)
                self.q_maxlen = tf.reduce_max(self.q_len)

                # 对context和question截取固定长度
                self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
                self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
                self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
                self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
                self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
                self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
                self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
                self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
            else:
                self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

            self.ch_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
            self.qh_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

            self.forward()
            total_params()

            if trainable:
                # 学习率
                self.lr = tf.minimum(config.learning_rate,
                                     0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
                # 优化器
                self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
                grads = self.opt.compute_gradients(self.loss)
                gradients, variables = zip(*grads)
                # 梯度截断
                capped_grads, _ = tf.clip_by_global_norm(
                    gradients, config.grad_clip)
                self.train_op = self.opt.apply_gradients(
                    zip(capped_grads, variables), global_step=self.global_step)

    def forward(self):
        config = self.config
        '''
        N: batch_size
        PL: passage长度
        QL: question长度
        CL: 单词最大字符长度
        d: 输出通道数
        dc: 字母的嵌入维度
        nh: 自注意力的头数
        '''
        N, PL, QL, CL, d, dc, nh = config.batch_size if not self.demo else 1, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.num_heads
        # Embedding层：获取词向量和字符向量的拼接
        with tf.variable_scope("Input_Embedding_Layer"):
            # # character嵌入：
            # 1、先对单词的每个字母进行word2vec
            ch_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, self.ch), [N * PL, CL, dc])
            qh_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, self.qh), [N * QL, CL, dc])
            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)

            # 2、将单词对应的word2vec矩阵通过conv编码成向量
            # 卷积
            ch_emb = conv(ch_emb, d,
                          bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=None)
            qh_emb = conv(qh_emb, d,
                          bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=True)
            # max_time_pooling
            ch_emb = tf.reduce_max(ch_emb, axis=1)
            qh_emb = tf.reduce_max(qh_emb, axis=1)

            ch_emb = tf.reshape(ch_emb, [N, PL, ch_emb.shape[-1]])
            qh_emb = tf.reshape(qh_emb, [N, QL, ch_emb.shape[-1]])

            # # 词嵌入：从glove获取
            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            # 拼接词向量和字符向量　
            # c_emb_size = [batch, n_c, c_emb+ch_emb]
            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            # q_emb_size = [batch, n_q, c_emb + ch_emb]
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

            # 分别通过highway网络
            # c_emb_size = [batch, n_c, d]
            c_emb = highway(c_emb, size=d, scope="highway", dropout=self.dropout, reuse=None)
            # c_emb_size = [batch, n_q, d]
            q_emb = highway(q_emb, size=d, scope="highway", dropout=self.dropout, reuse=True)

        # Stacking Embedding Encoder Block的实现：共1个encoder block，每个7个卷积层，卷积核数d=96
        with tf.variable_scope("Embedding_Encoder_Layer"):
            # c_size = [batch, n_c, d]
            c = residual_block(c_emb,
                               num_blocks=1,
                               num_conv_layers=4,
                               kernel_size=7,
                               mask=self.c_mask,
                               num_filters=d,
                               num_heads=nh,
                               seq_len=self.c_len,
                               scope="Encoder_Residual_Block",
                               bias=False,
                               dropout=self.dropout)
            # q_size = [batch, n_q, d]
            q = residual_block(q_emb,
                               num_blocks=1,
                               num_conv_layers=4,
                               kernel_size=7,
                               mask=self.q_mask,
                               num_filters=d,
                               num_heads=nh,
                               seq_len=self.q_len,
                               scope="Encoder_Residual_Block",
                               reuse=True,  # 共享passage和question的Stacking Embedding Encoder Block的权重
                               bias=False,
                               dropout=self.dropout)

        # Context-Query-Attention实现：
        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            # C = tf.tile(tf.expand_dims(c,2),[1,1,self.q_maxlen,1])
            # Q = tf.tile(tf.expand_dims(q,1),[1,self.c_maxlen,1,1])
            # S = trilinear([C, Q, C*Q], input_keep_prob = 1.0 - self.dropout)
            # S_size = [batch, n_c, n_q], q_size = [batch, n_q, d], c_size = [batch, n_c, d]
            S = optimized_trilinear_for_attention([c, q], self.c_maxlen, self.q_maxlen,
                                                  input_keep_prob=1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            # n_q方向进行softmax
            S_ = tf.nn.softmax(mask_logits(S, mask=mask_q), dim=-1)
            mask_c = tf.expand_dims(self.c_mask, 2)
            # n_c方向进行softmax
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask=mask_c), dim=1), (0, 2, 1))
            # c2q_size = [batch, n_c, d]
            self.c2q = tf.matmul(S_, q)
            # q2c_size = [batch, n_c, d]
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c)
            # attention_size = [4, batch, n_c, d]
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]

        # Stacked Model Encoder Blocks实现：共7个encoder block，每个2个卷积层，卷积核数d=96
        with tf.variable_scope("Model_Encoder_Layer"):
            # c, self.c2q, c * self.c2q, c * self.q2c 按照通道维度进行合并
            inputs = tf.concat(attention_outputs, axis=-1)
            self.enc = [conv(inputs, d, name="input_projection")]
            # 3个Stacked Model Encoder Blocks
            for i in range(3):
                if i % 2 == 0:  # 每两层进行一次dropout
                    self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
                self.enc.append(
                    residual_block(self.enc[i],
                                   num_blocks=7,
                                   num_conv_layers=2,
                                   kernel_size=5,
                                   mask=self.c_mask,
                                   num_filters=d,
                                   num_heads=nh,
                                   seq_len=self.c_len,
                                   scope="Model_Encoder",
                                   bias=False,
                                   reuse=True if i > 0 else None,  # 共享同一个Stacked Model Encoder Blocks的权重
                                   dropout=self.dropout)
                )

        # 输出层实现：
        with tf.variable_scope("Output_Layer"):
            # 合并Stacked Model Encoder Blocks的第一个和第二个输出，并和并通道
            start_logits = tf.squeeze(
                conv(tf.concat([self.enc[1], self.enc[2]], axis=-1), 1, bias=False, name="start_pointer"), -1)
            # 合并Stacked Model Encoder Blocks的第一个和第三个输出，并和并通道
            end_logits = tf.squeeze(
                conv(tf.concat([self.enc[1], self.enc[3]], axis=-1), 1, bias=False, name="end_pointer"), -1)

            self.logits = [mask_logits(start_logits, mask=self.c_mask),
                           mask_logits(end_logits, mask=self.c_mask)]

            logits1, logits2 = [l for l in self.logits]

            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, config.ans_limit)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits1, labels=self.y1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits2, labels=self.y2)
            self.loss = tf.reduce_mean(losses + losses2)

        # L2正则化
        if config.l2_norm is not None:
            variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
            self.loss += l2_loss

        if config.decay is not None:
            self.var_ema = tf.train.ExponentialMovingAverage(config.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)

                self.assign_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v:
                        self.assign_vars.append(tf.assign(var, v))

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
