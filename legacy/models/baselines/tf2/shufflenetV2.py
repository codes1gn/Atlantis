import tensorflow as tf
import tf_slim as ts
import argparse
import numpy as np
from shared_functions import make_activation, make_conv2d, make_conv2d_bn, measure_tf2_gpu, get_tf2_args

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers
import time


def batch_norm(training):
    return layers.BatchNormalization(axis=3, momentum=0.99,
                                     epsilon=0.001, center=True,
                                     scale=True, trainable=training, fused=True)


class ShufflenetV2(tf.keras.Model):
    def __init__(self, num_classes, training):
        super(ShufflenetV2, self).__init__()

        self.training = training
        self.num_classes = num_classes

        self.conv1 = layers.Conv2D(
            24, kernel_size=3, strides=2, padding='SAME')
        self.bn1 = batch_norm(self.training)
        self.act1 = layers.Activation("relu")
        self.maxpool1 = layers.MaxPooling2D((3, 3), (2, 2), padding='SAME')

        self.block1 = ShuffleBlock(
            num_units=4, in_channels=24, out_channels=116)
        self.block2 = ShuffleBlock(num_units=8, in_channels=116)
        self.block3 = ShuffleBlock(num_units=4, in_channels=232)

        self.globalavgpool = layers.GlobalAveragePooling2D()
        self.dropout1 = layers.Dropout(rate=0.7)
        self.dense1 = layers.Dense(1)

        self.conv5 = layers.Conv2D(
            1024, kernel_size=1, strides=1, padding='SAME')
        self.bn5 = batch_norm(self.training)
        self.act5 = layers.Activation("relu")

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)

        x = self.globalavgpool(x)
        x = self.dropout1(x, training=self.training)
        x = self.dense1(x)

        return x


class ShuffleBlock(tf.keras.layers.Layer):
    def __init__(self, num_units, in_channels, out_channels=None, training=True):
        super(ShuffleBlock, self).__init__()

        self.training = training
        self.num_units = num_units
        self.in_channels = in_channels
        self.out_channels = 2 * self.in_channels if out_channels is None else out_channels

        self.all_basic_uint = []
        for j in range(2, self.num_units + 1):
            self.all_basic_uint.append(
                BasicUnit(in_channels=self.out_channels//2))

        self.conv1 = layers.Conv2D(
            self.in_channels, kernel_size=1, strides=1, padding='SAME')
        self.bn1 = batch_norm(self.training)
        self.act1 = layers.Activation("relu")

        self.dwconv2 = layers.DepthwiseConv2D(
            kernel_size=3, strides=2, padding='SAME')
        self.bn2 = batch_norm(self.training)

        self.conv3 = layers.Conv2D(
            self.out_channels // 2, kernel_size=1, strides=1, padding='SAME')
        self.bn3 = batch_norm(self.training)
        self.act3 = layers.Activation("relu")

        self.dwconv4 = layers.DepthwiseConv2D(
            kernel_size=3, strides=2, padding='SAME')
        self.bn4 = batch_norm(self.training)

        self.conv5 = layers.Conv2D(
            self.out_channels // 2, kernel_size=1, strides=1, padding='SAME')
        self.bn5 = batch_norm(self.training)
        self.act5 = layers.Activation("relu")

    def shuffle_xy(self, x, y):
        batch_size, height, width, channels = x.shape[:]
        depth = channels
        z = tf.stack([x, y], axis=3)
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [batch_size, height, width, 2*depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y

    def call(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)

        y = self.dwconv2(y)
        y = self.bn2(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = self.act3(y)

        x = self.dwconv4(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)

        basic_uint_count = 0
        for j in range(2, self.num_units + 1):
            x, y = self.shuffle_xy(x, y)
            x = self.all_basic_uint[basic_uint_count](x)
            basic_uint_count += 1

        x = layers.concatenate([x, y])
        return x


class BasicUnit(tf.keras.layers.Layer):
    def __init__(self, in_channels=10, training=True):
        super(BasicUnit, self).__init__()
        self.in_channels = in_channels

        self.training = training
        self.conv1 = layers.Conv2D(
            self.in_channels, kernel_size=1, strides=1, padding='SAME')
        self.bn1 = batch_norm(self.training)
        self.act1 = layers.Activation("relu")

        self.dwconv2 = layers.DepthwiseConv2D(
            kernel_size=3, strides=1, padding='SAME')
        self.bn2 = batch_norm(self.training)

        self.conv3 = layers.Conv2D(
            self.in_channels, kernel_size=1, strides=1, padding='SAME')
        self.bn3 = batch_norm(self.training)
        self.act3 = layers.Activation("relu")

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.dwconv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        return x


# @tf.function(jit_compile=False)
@tf.function(experimental_compile=False)
def shufflenetV2(input):
    return ShufflenetV2(num_classes=1, training=False)

# @tf.function(jit_compile=True)


@tf.function(experimental_compile=True)
def shufflenetV2(input):
    return ShufflenetV2(num_classes=1, training=False)


if __name__ == '__main__':
    # parse tf2 arguments
    args = get_tf2_args()

    args.network = 'shufflenetV2'
    input_shape = (args.batch_size, 3, 224, 224)
    inputs = np.random.uniform(-1, 1, size=input_shape).astype("float32")

    method_name = 'TF'
    measure_tf2_gpu(shufflenetV2, inputs, method_name, args)

    # This errors out; resize kernel is not supported even by the most recent XLA
    method_name = 'TF-XLA'
    measure_tf2_gpu(shufflenetV2, inputs, method_name, args)
