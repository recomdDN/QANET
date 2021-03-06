# -*- coding: utf-8 -*-
# /usr/bin/python2

import tensorflow as tf
import numpy as np
import math

from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import RNNCell

from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops

from functools import reduce
from operator import mul

'''
Some functions are taken directly from Tensor2Tensor Library:
https://github.com/tensorflow/tensor2tensor/
and BiDAF repository:
https://github.com/allenai/bi-att-flow
'''

initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                     mode='FAN_AVG',
                                                                     uniform=True,
                                                                     dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                          mode='FAN_IN',
                                                                          uniform=False,
                                                                          dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)


def glu(x):
    """Gated Linear Units from https://arxiv.org/pdf/1612.08083.pdf"""
    x, x_h = tf.split(x, 2, axis=-1)
    return tf.sigmoid(x) * x_h


def noam_norm(x, epsilon=1.0, scope=None, reuse=None):
    """One version of layer normalization."""
    with tf.name_scope(scope, default_name="noam_norm", values=[x]):
        shape = x.get_shape()
        ndims = len(shape)
        return tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) * tf.sqrt(tf.to_float(shape[-1]))


def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    # 按通道求均值 shape = [batch, h, w, 1]
    mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
            "layer_norm_scale", [filters], regularizer=regularizer, initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [filters], regularizer=regularizer, initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result


norm_fn = layer_norm  # tf.contrib.layers.layer_norm #tf.contrib.layers.layer_norm or noam_norm


def highway(x, size=None, activation=None,
            num_layers=2, scope="highway", dropout=0.0, reuse=None):
    """
    conv-highway网络，使用卷积核大小为1（不改变输入张量的width）的卷积实现了highway网络
    :param x: 输入张量，shape=[batches, lengths, channels]
    :param size: 输出通道数
    :param activation: 激活函数
    :param num_layers: highway网络的层数
    :param scope: 命名空间名
    :param dropout: dropout比例
    :param reuse: 是否参数重用
    :return: 输出张量，shape=[batches, lengths, size]
    """
    with tf.variable_scope(scope, reuse):
        # 如果没指定输出通道数，那么输入通道数与输出通道数相等
        if size is None:
            size = x.shape.as_list()[-1]
        # 如果有指定输出通道数，那么先对输入通道进行压缩，压缩到输出通道数相同，
        # 否则x * (1 - sigmoid(conv2(x)))由于通道数不同而无法计算
        else:
            x = conv(x, size, name="input_projection", reuse=reuse, kernel_size=1)
        # conv1(x) * sigmoid(conv2(x)) + x * (1 - sigmoid(conv2(x)))
        for i in range(num_layers):
            T = conv(x, size, bias=True, activation=tf.sigmoid,
                     name="gate_%d" % i, reuse=reuse)
            H = conv(x, size, bias=True, activation=activation,
                     name="activation_%d" % i, reuse=reuse)
            H = tf.nn.dropout(H, 1.0 - dropout)
            x = H * T + x * (1.0 - T)
        return x

# 层dropout
def layer_dropout(inputs, residual, dropout):
    pred = tf.random_uniform([]) < dropout
    return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)

# 包含残差卷积层，残差自注意力层，残差前向网络层
def residual_block(inputs, num_blocks, num_conv_layers, kernel_size, mask=None,
                   num_filters=128, input_projection=False, num_heads=8,
                   seq_len=None, scope="res_block", is_training=True,
                   reuse=None, bias=True, dropout=0.0):
    with tf.variable_scope(scope, reuse=reuse):
        if input_projection:
            inputs = conv(inputs, num_filters, name="input_projection", reuse=reuse, kernel_size=1)
        outputs = inputs
        sublayer = 1
        total_sublayers = (num_conv_layers + 2) * num_blocks
        # 添加num_blocks个Encoder Block
        for i in range(num_blocks):
            # 添加位置信息
            outputs = add_timing_signal_1d(outputs)
            # 卷积残差模块
            outputs, sublayer = conv_block(outputs, num_conv_layers, kernel_size, num_filters,
                                           seq_len=seq_len, scope="encoder_block_%d" % i, reuse=reuse, bias=bias,
                                           dropout=dropout, sublayers=(sublayer, total_sublayers))
            # 注意力残差模块
            outputs, sublayer = self_attention_block(outputs, num_filters, seq_len, mask=mask, num_heads=num_heads,
                                                     scope="self_attention_layers%d" % i, reuse=reuse,
                                                     is_training=is_training,
                                                     bias=bias, dropout=dropout, sublayers=(sublayer, total_sublayers))
        return outputs

# 残差卷积模块
def conv_block(inputs, num_conv_layers, kernel_size, num_filters,
               seq_len=None, scope="conv_block", is_training=True,
               reuse=None, bias=True, dropout=0.0, sublayers=(1, 1)):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.expand_dims(inputs, 2)
        l, L = sublayers
        for i in range(num_conv_layers):
            residual = outputs
            # 先进行Layernorm
            outputs = norm_fn(outputs, scope="layer_norm_%d" % i, reuse=reuse)
            # 每两层进行一次dropout
            if (i) % 2 == 0:
                outputs = tf.nn.dropout(outputs, 1.0 - dropout)
            # 进行深宽分离卷积
            outputs = depthwise_separable_convolution(outputs,
                                                      kernel_size=(kernel_size, 1), num_filters=num_filters,
                                                      scope="depthwise_conv_layers_%d" % i, is_training=is_training,
                                                      reuse=reuse)
            # 层dropout
            outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
            l += 1
        return tf.squeeze(outputs, 2), l

# 自注意力残差模块+前向网络
def self_attention_block(inputs, num_filters, seq_len, mask=None, num_heads=8,
                         scope="self_attention_ffn", reuse=None, is_training=True,
                         bias=True, dropout=0.0, sublayers=(1, 1)):
    with tf.variable_scope(scope, reuse=reuse):
        l, L = sublayers
        # 1、先进行Layernorm，dropout，多头自注意力，层dropout
        outputs = norm_fn(inputs, scope="layer_norm_1", reuse=reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = multihead_attention(outputs, num_filters,
                                      num_heads=num_heads, seq_len=seq_len, reuse=reuse,
                                      mask=mask, is_training=is_training, bias=bias, dropout=dropout)
        residual = layer_dropout(outputs, inputs, dropout * float(l) / L)
        l += 1

        # 2、再进行Layernorm，dropout，两层前向网络，层droupout
        outputs = norm_fn(residual, scope="layer_norm_2", reuse=reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        #
        outputs = conv(outputs, num_filters, True, tf.nn.relu, name="FFN_1", reuse=reuse, kernel_size=1)
        outputs = conv(outputs, num_filters, True, None, name="FFN_2", reuse=reuse, kernel_size=1)
        outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
        l += 1
        return outputs, l

# 多头注意力模块，input_size = [batch, seq_len, channel]
def multihead_attention(queries, units, num_heads,
                        memory=None,
                        seq_len=None,
                        scope="Multi_Head_Attention",
                        reuse=None,
                        mask=None,
                        is_training=True,
                        bias=True,
                        dropout=0.0):
    """实现方式： 1、先对queries和memory进行通道压缩，压缩为depth
                2、再对步骤1重复num_heads次
                3、结合1、2即对queries和memory通道压缩为depth * num_heads = units的卷积
    :param queries: 输入张量Q
    :param units: int，输出通道数，num_heads的整数倍
    :param num_heads: int，head的个数
    :param memory: 输入张量KV
    :param seq_len: 序列长度
    :param scope: 命名空间名
    :param reuse: 是否复用参数
    :param mask:
    :param is_training: 是否可训练
    :param bias: 是否需要偏置
    :param dropout: dropout比例
    :return: 输出张量，shape = [batch, seq_len, units]
    """
    with tf.variable_scope(scope, reuse=reuse):
        # 自注意力
        if memory is None:
            memory = queries

        memory = conv(memory, 2 * units, name="memory_projection", reuse=reuse, kernel_size=1)
        query = conv(queries, units, name="query_projection", reuse=reuse, kernel_size=1)
        # Q,K,V 原来shape = [batch, length_q, units] 相当于head==1
        # Q,K,V 现在shape = [batch, num_heads, length_q, depth_k] 相当与head==num_heads
        # multi-head Q、K、V
        Q = split_last_dimension(query, num_heads)
        # memory按通道半分获得K，V
        K, V = [split_last_dimension(tensor, num_heads) for tensor in tf.split(memory, 2, axis=2)]

        key_depth_per_head = units // num_heads
        Q *= key_depth_per_head ** -0.5
        x = dot_product_attention(Q, K, V,
                                  bias=bias,
                                  seq_len=seq_len,
                                  mask=mask,
                                  is_training=is_training,
                                  scope="dot_product_attention",
                                  reuse=reuse, dropout=dropout)
        return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))

# 卷积模块
def conv(inputs, output_size, bias=None, activation=None, kernel_size=1, name="conv", reuse=None):
    """
    卷积运算，默认卷积核大小为1，卷积只改变in_channels，不改变新in_widths
    :param inputs: 输入张量，shape可以为[batch, 1, in_width, in_channels]
                ，也可以为[batch, in_width, in_channels]
    :param output_size: int，输出通道数
    :param bias: bool，是否使用偏置
    :param activation: 激活函数
    :param kernel_size: int，卷积核大小
    :param name: string，命名空间
    :param reuse: bool，是否复用参数
    :return: 输出张量，对应shape为[batch, 1, in_width-kernel_size+1, output_size]
            或者[batch, in_width-kernel_size+1, output_size]
    """
    with tf.variable_scope(name, reuse=reuse):
        shapes = inputs.shape.as_list()

        # 输入维度>4时报错
        if len(shapes) > 4:
            raise NotImplementedError
        # 输入维度为4时，使用2维卷积的参数
        # inputs_size = [batch, in_height, in_width, in_channels]
        elif len(shapes) == 4:
            # [filter_height, filter_width, in_channels, out_channels]
            filter_shape = [1, kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, 1, output_size]
            # The stride of the sliding window for each dimension of `input`.
            strides = [1, 1, 1, 1]
        # 输入维度为3时，使用1维卷积的参数
        # inputs_size = [batch, in_width, in_channels]
        else:
            # [filter_width, in_channels, out_channels]
            filter_shape = [kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, output_size]
            strides = 1
        # 根据输入数据的维度来判断使用2维卷积还是1维卷积
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                                  filter_shape,
                                  dtype=tf.float32,
                                  regularizer=regularizer,
                                  initializer=initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        # 是否使用bias
        if bias:
            outputs += tf.get_variable("bias_",
                                       bias_shape,
                                       regularizer=regularizer,
                                       initializer=tf.zeros_initializer())
        # 是否使用激活函数
        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def mask_logits(inputs, mask, mask_value=-1e30):
    shapes = inputs.shape.as_list()
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)

# 深宽分离卷积 # [filter_height, filter_width, in_channels, out_channels]
# 卷积方法为same，即不改变输入形状，只改变通道数
def depthwise_separable_convolution(inputs, kernel_size, num_filters,
                                    scope="depthwise_separable_convolution",
                                    bias=True, is_training=True, reuse=None):
    """
    深度和宽度分离卷积，默认是用same卷积，输出不改变in_width，只改变in_channels
    :param inputs: 输入张量，shape=[batch, in_width, in_channels]
    :param kernel_size: int，卷积和大小
    :param num_filters: int，输出通道数
    :param scope: string，命名空间名
    :param bias: bool，是否要偏置
    :param is_training: bool，参数是否能训练
    :param reuse: bool，参数是否复用
    :return: 输出张量，shape=[batch, in_width, num_filters]
    """
    with tf.variable_scope(scope, reuse=reuse):
        shapes = inputs.shape.as_list()
        # [filter_height, filter_width, in_channels, channel_multiplier]
        depthwise_filter = tf.get_variable("depthwise_filter",
                                           (kernel_size[0], kernel_size[1], shapes[-1], 1),
                                           dtype=tf.float32,
                                           regularizer=regularizer,
                                           initializer=initializer_relu())
        # [1, 1, channel_multiplier * in_channels, out_channels]`
        pointwise_filter = tf.get_variable("pointwise_filter",
                                           (1, 1, shapes[-1], num_filters),
                                           dtype=tf.float32,
                                           regularizer=regularizer,
                                           initializer=initializer_relu())
        outputs = tf.nn.separable_conv2d(inputs,
                                         depthwise_filter,
                                         pointwise_filter,
                                         strides=(1, 1, 1, 1),
                                         padding="SAME")
        if bias:
            b = tf.get_variable("bias",
                                outputs.shape[-1],
                                regularizer=regularizer,
                                initializer=tf.zeros_initializer())
            outputs += b
        outputs = tf.nn.relu(outputs)
        return outputs


def split_last_dimension(x, n):
    """对最后一个维度进行切分，切分的第一个维度为n
    :param x: 输入张量，shape = [..., m]
    :param n: int
    :return: 输出张量，shape = [..., n, m/n]
    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return tf.transpose(ret, [0, 2, 1, 3])

# 点积注意力层
def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          seq_len=None,
                          mask=None,
                          is_training=True,
                          scope=None,
                          reuse=None,
                          dropout=0.0):
    """dot-product attention.
    :param q: a Tensor with shape [batch, heads, length_q, depth_k]
    :param k: a Tensor with shape [batch, heads, length_kv, depth_k]
    :param v: a Tensor with shape [batch, heads, length_kv, depth_v]
    :param bias: 是否添加偏置
    :param seq_len: 序列的长度
    :param mask:
    :param is_training: bool，参数是否可训练
    :param scope: 命名空间名
    :param reuse: 参数是否复用
    :param dropout: dropout比例
    :return:
    """
    with tf.variable_scope(scope, default_name="dot_product_attention", reuse=reuse):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias:
            b = tf.get_variable("bias",
                                logits.shape[-1],
                                regularizer=regularizer,
                                initializer=tf.zeros_initializer())
            logits += b
        if mask is not None:
            shapes = [x if x != None else -1 for x in logits.shape.as_list()]
            mask = tf.reshape(mask, [shapes[0], 1, 1, shapes[-1]])
            logits = mask_logits(logits, mask)
        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, 1.0 - dropout)
        return tf.matmul(weights, v)

# 拼接最后两个维度
def combine_last_two_dimensions(x):
    """合并输入张量x的维度，使得最后两个维度合并为一个维度
    :param x: 输入张量x，shape = [..., a, b]
    :return:  输出张量，shape = [..., ab]
    """
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal

# 获取位置信息
def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """PE(pos,2i) = sin(pos/10000^(2i/channel))
       PE(pos,2i+1) = cos(pos/10000^(2i/channel))
       也就是说给定pos，我们可以把他编码成一个channel的向量，位置编码的每一个维度
       对应正弦曲线，波长构成了从2pi到10000*2pi的等比数列。
    :param length: int， 时间序列的长度
    :param channels: int， 时间序列嵌入向量的长度
    :param min_timescale: float
    :param max_timescale: float
    :return: 输出张量，shape = [1, length, channels]
    """
    # 将position先编号，再把编号转换成为一个向量
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    # PE matrix
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    # 如果channels为基数，那么signal最后一列为全0
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def ndim(x):
    """以整数形式返回张量中的轴数。
    :param x: 输入张量x
    :return: int，表示输入张量的轴数
    """
    dims = x.get_shape().dims
    if dims is not None:
        return len(dims)
    return None


def dot(x, y):
    """张量内积,把x,y看作一系列维度为shape(x)[-1]==shape(y)[-2]的向量，然后计算向量内积
    将x reshape为[-1,shape(x)[-1]], 将y reshape为[shape(y)[-2],-1],然后tf.matmul(x,y)
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)
    :param x: 输入张量x
    :param y: 输入张量y
    :return: 输出张量
    """
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        # 存放x的shape
        x_shape = []
        for i, s in zip(x.get_shape().as_list(), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        # 存放y的shape
        y_shape = []
        for i, s in zip(y.get_shape().as_list(), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        # 每个轴的顺序
        y_permute_dim = list(range(ndim(y)))
        # 把倒数第二个轴换到第一个轴上
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        # reshape成二维矩阵
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        # 矩阵乘法后进行reshape
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if isinstance(x, tf.SparseTensor):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out

# 相当与tf.matmul的高级版，可以指定做内积的轴
def batch_dot(x, y, axes=None):
    """Copy from keras==2.0.6
    Batchwise dot product.

    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.

    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.
    """
    # 如果axes只指定一个整型值，那么认为axes[0]=axes[1]
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    # 如果x,y轴数不等，就给轴数少的增加轴
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    # 如果x的轴数小于y的轴数
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    # 相等
    else:
        diff = 0
    #  x的轴数与y的轴数都等于2
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    # x,y至少一个轴数大于二
    else:
        # axes不为None
        if axes is not None:
            # 判断指定的轴是否为最后一个轴，如果axes[0]为x的倒数第二个轴，则x需要转置，如果axes[1]为y的最后一个轴，则y需要转置
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    # 如果x,y轴数不等
    if diff:
        # x轴数大于y
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = tf.expand_dims(out, 1)
    return out

# 三次线性函数
def optimized_trilinear_for_attention(args, c_maxlen, q_maxlen, input_keep_prob=1.0,
                                      scope='efficient_trilinear',
                                      bias_initializer=tf.zeros_initializer(),
                                      kernel_initializer=initializer()):
    # args参数个数不等于二则报错
    assert len(args) == 2, "just use for computing attention with two input"
    # 分别获取两个参数的shape
    # arg0_shape = [batch, len_c, dimension]
    arg0_shape = args[0].get_shape().as_list()
    # arg0_shape = [batch, len_q, dimension]
    arg1_shape = args[1].get_shape().as_list()
    # 如果两个参数轴数不为3则报错
    if len(arg0_shape) != 3 or len(arg1_shape) != 3:
        raise ValueError("`args` must be 3 dims (batch_size, seq_len, dimension)")
    # 如果两个参数最后一个维度不同则报错
    if arg0_shape[2] != arg1_shape[2]:
        raise ValueError("the last dimension of `args` must equal")
    arg_size = arg0_shape[2]
    dtype = args[0].dtype
    # 对c，q使用dropout
    droped_args = [tf.nn.dropout(arg, input_keep_prob) for arg in args]
    with tf.variable_scope(scope):
        # [dimension, 1]
        weights4arg0 = tf.get_variable(
            "linear_kernel4arg0", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        # [dimension, 1]
        weights4arg1 = tf.get_variable(
            "linear_kernel4arg1", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        # [1, 1, dimension]
        weights4mlu = tf.get_variable(
            "linear_kernel4mul", [1, 1, arg_size],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        biases = tf.get_variable(
            "linear_bias", [1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=bias_initializer)
        # subres0_shape = [batch, len_c, q_maxlen]
        subres0 = tf.tile(dot(droped_args[0], weights4arg0), [1, 1, q_maxlen])
        # subres1_shape = [batch, c_maxlen, len_q]
        subres1 = tf.tile(tf.transpose(dot(droped_args[1], weights4arg1), perm=(0, 2, 1)), [1, c_maxlen, 1])
        # subres2_shape = [bacth, len_c, len_q]
        subres2 = batch_dot(droped_args[0] * weights4mlu, tf.transpose(droped_args[1], perm=(0, 2, 1)))
        res = subres0 + subres1 + subres2
        nn_ops.bias_add(res, biases)
        return res

# 原三次线性实现
def trilinear(args,
              output_size=1,
              bias=True,
              squeeze=False,
              wd=0.0,
              input_keep_prob=1.0,
              scope="trilinear"):
    with tf.variable_scope(scope):
        # args = [C, Q, C*Q]
        flat_args = [flatten(arg, 1) for arg in args]
        flat_args = [tf.nn.dropout(arg, input_keep_prob) for arg in flat_args]
        flat_out = _linear(flat_args, output_size, bias, scope=scope)
        out = reconstruct(flat_out, args[0], 1)
        return tf.squeeze(out, -1)


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep):
    # ref,tensor的shape
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out


def _linear(args,
            output_size,
            bias,
            bias_initializer=tf.zeros_initializer(),
            scope=None,
            kernel_initializer=initializer(),
            reuse=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    :param args: args是一个二维张量或者是一个二维张量的列表
    :param output_size: int，W[i]的第二维度
    :param bias: bool，是否添加偏置
    :param bias_initializer: 偏置初始化函数
    :param scope: 命名空间名
    :param kernel_initializer: 卷积核参数初始化函数
    :param reuse: 参数是否复用
    :return: 二维张量，shape = [batch x output_size] = sum_i(args[i] * W[i])
  """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]
    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # W[q, c, q*c]
    with tf.variable_scope(scope, reuse=reuse) as outer_scope:
        weights = tf.get_variable(
            "linear_kernel", [total_arg_size, output_size],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        # 是否添加偏置
        if not bias:
            return res
        with tf.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = tf.get_variable(
                "linear_bias", [output_size],
                dtype=dtype,
                regularizer=regularizer,
                initializer=bias_initializer)
        return nn_ops.bias_add(res, biases)

# 统计模型参数个数
def total_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))
