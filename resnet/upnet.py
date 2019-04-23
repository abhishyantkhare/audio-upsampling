import os
import random

import tensorflow as tf
from tensorflow.python.keras import Model, Input

from tensorflow.python.keras.layers import Conv1D, Dropout, LeakyReLU, Activation, merge

INPUT_SAMPLE_RATE = 1000
OUTPUT_SAMPLE_RATE = 22050
SAMPLE_LENGTH = 0.5

# Network design modeled after
# https://github.com/kuleshov/audio-super-res


def SubPixel1D(I, r):
    """One-dimensional subpixel upsampling layer

    Calls a tensorflow function that directly implements this functionality.
    We assume input has dim (batch, width, r)
    """
    with tf.name_scope('subpixel'):
        X = tf.transpose(I, [2, 1, 0])  # (r, w, b)
        X = tf.batch_to_space_nd(X, [r], [[0, 0]])  # (1, r*w, b)
        X = tf.transpose(X, [2, 1, 0])
        return X


class UpNet:
    def __init__(self, input_length, output_length, num_layers=4,
                 batch_size=128, learning_rate=1e-4, b1=0.99, b2=0.999):
        self.r = 2
        self.layers = num_layers

        self.input = Input(shape=(batch_size, input_length, 1))
        x = self.input

        with tf.name_scope('generator'):
            n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
            n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]
            downsampling_l = []

            print('Generating model')

            # Downsampling layers
            for l, nf, fs in zip(range(num_layers), n_filters, n_filtersizes):
                with tf.name_scope('downsc_conv%d' % l):
                    x = Conv1D(nf, fs, input_shape=x.shape.as_list()[1:])(x)
                    x = LeakyReLU(0.2)(x)
                    print('D-Block: ', x.get_shape(), downsampling_l.append(x))

            # bottleneck layer
            with tf.name_scope('bottleneck_conv'):
                x = (Conv1D(n_filters[-1], n_filtersizes[-1]))(x)
                x = Dropout(0.5)(x)
                # x = BatchNormalization(mode=2)(x)
                x = LeakyReLU(0.2)(x)

            # upsampling layers
            for l, nf, fs, l_in in reversed(list(zip(
                    range(num_layers), n_filters, n_filtersizes, downsampling_l
            ))):
                with tf.name_scope('upsc_conv%d' % l):
                    # (-1, n/2, 2f)
                    x = (Conv1D(2 * nf, fs))(x)
                    # x = BatchNormalization(mode=2)(x)
                    x = Dropout(0.5)(x)
                    x = Activation('relu')(x)
                    # (-1, n, f)
                    x = SubPixel1D(x, 2)
                    # (-1, n, 2f)
                    x = merge.concatenate([x, l_in])
                    print('U-Block: ', x.get_shape())

            # final conv layer
            with tf.name_scope('lastconv'):
                x = Conv1D(2, 9)(x)
                x = SubPixel1D(x, 2)
                print(x.get_shape())

            self.output = merge.add([x, self.input])
            self.model = Model(inputs=x, outputs=self.output)


def train(epochs=1, model_name=None):
    if model_name is None:
        # Initialize model
        upnet = UpNet(
            int(INPUT_SAMPLE_RATE * SAMPLE_LENGTH),
            int(OUTPUT_SAMPLE_RATE * SAMPLE_LENGTH)
        )
    else:
        # Load model from file
        upnet = None
    # Initialize list of available data
    input_directory = '../data/wav_{}/'.format(INPUT_SAMPLE_RATE)
    output_directory = '../data/wav_{}/'.format(OUTPUT_SAMPLE_RATE)
    input_files = [input_directory + fn for fn in os.listdir(input_directory)]
    output_files = [output_directory + fn for fn in os.listdir(output_directory)]
    assert len(input_files) == len(output_files)
    assert all([fn.endswith('.wav') for fn in input_files + output_files])
    pairs = list(zip(input_files, output_files))
    random.seed(0)
    random.shuffle(pairs)
    # Train
    for _ in range(epochs):
        upnet.model.compile(
            optimizer='adam', loss='mean_squared_error', metrics=['accuracy']
        )


def eval():
    pass
