import argparse
import tensorflow as tf
import numpy as np
import time
import timeit
from shared_functions import make_activation, make_conv2d, make_conv2d_bn, measure_tf2_gpu, get_tf2_args

# tf.config.run_functions_eagerly(False)
#
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)


def resnet_block(input, strides, out_channels, name):
    t = make_conv2d(input_tensor=input, filter_shape=(1, 1, input.shape[1], out_channels), strides=(
        1, 1, 1, 1), padding="SAME", actimode="RELU", name=name+"_conv1")
    t = make_conv2d(input_tensor=t, filter_shape=(3, 3, out_channels, out_channels),
                    strides=strides, padding="SAME", actimode="RELU", name=name+"_conv2")
    t = make_conv2d(input_tensor=t, filter_shape=(1, 1, out_channels, out_channels*4),
                    strides=(1, 1, 1, 1), padding="SAME", actimode="NONE", name=name+"_conv3")
    if (strides[2] > 1) or (input.shape[1] != out_channels * 4):
        input = make_conv2d(input_tensor=input, filter_shape=(
            1, 1, input.shape[1], out_channels*4), strides=strides, padding="SAME", actimode="RELU", name=name+"_conv4")
    return tf.nn.relu(tf.add(input, t))


def resnet50_tf2_model(input):

    t = input

    if (input[1] == input[2]):
        strides = (1, 2, 2, 1)
    else:
        strides = (1, 1, 2, 2)
    for i in range(3):
        t = resnet_block(t, (1, 1, 1, 1), 64, "resnet_block_1_{}".format(i))

    if (input[1] == input[2]):
        strides = (1, 2, 2, 1)
    else:
        strides = (1, 1, 2, 2)
    for i in range(4):
        t = resnet_block(t, strides, 128, "resnet_block_2_{}".format(i))
        strides = (1, 1, 1, 1)

    if (input[1] == input[2]):
        strides = (1, 2, 2, 1)
    else:
        strides = (1, 1, 2, 2)
    for i in range(6):
        t = resnet_block(t, strides, 256, "resnet_block_3_{}".format(i))
        strides = (1, 1, 1, 1)

    if (input[1] == input[2]):
        strides = (1, 2, 2, 1)
    else:
        strides = (1, 1, 2, 2)
    for i in range(3):
        t = resnet_block(t, strides, 512, "resnet_block_4_{}".format(i))
        strides = (1, 1, 1, 1)
    return t

# @tf.function(jit_compile=False)


@tf.function(experimental_compile=False)
def resnet50_tf2(input):
    return resnet50_tf2_model(input)

# @tf.function(jit_compile=True)


@tf.function(experimental_compile=True)
def resnet50_tf2_xla(input):
    return resnet50_tf2_model(input)


if __name__ == '__main__':
    # parse tf2 arguments
    args = get_tf2_args()

    args.network = 'resnet50'
    if args.hardware == 'gpu':
        input_shape = (args.batch_size, 64, 56, 56)
    elif args.hardware == 'cpu':
        input_shape = (args.batch_size, 56, 56, 64)
    inputs = np.random.uniform(-1, 1, size=input_shape).astype("float32")

    method_name = 'TF'
    measure_tf2_gpu(resnet50_tf2, inputs, method_name, args)

    # This errors out; resize kernel is not supported even by the most recent XLA
    method_name = 'TF-XLA'
    measure_tf2_gpu(resnet50_tf2_xla, inputs, method_name, args)
