import os
import random

import torch
import torch.nn as nn

from scipy.io import wavfile
import numpy as np
from fileserver import Fileserver
from subprocess import call

INPUT_SAMPLE_RATE = 1000
OUTPUT_SAMPLE_RATE = 2000
SAMPLE_LENGTH = 1
ROOTDIR = '/Volumes/fileserver.brianlevis.com/BRIANDISK/tensorpros/fma_small/'



DTYPE_RANGES = {
    np.float32: (-1.0, 1.0), np.int32: (-2147483648, 2147483647), 
    np.dtype('int16'): (-32768, 32767), np.dtype('uint8'): (0, 255)
}
BITS_TO_DTYPE = {
    64: np.float32, 32: np.int32, 16: np.dtype('int16'), 8: np.dtype('uint8')
}


# Network design modeled after
# https://github.com/kuleshov/audio-super-res


def SubPixel1D(I, r):
    """One-dimensional subpixel upsampling layer

    Calls a tensorflow function that directly implements this functionality.
    We assume input has dim (batch, width, r)
    """
      b , w, r = I.size()
      X = X.permute(2, 1, 0)  # (r, w, b)
      X = X.view(1, r*w, b)  # (1, r*w, b)
      X = X.permute(2, 1, 0)
      return X

def load_raw_input(fname, fs):
  # Reduce bitrate of audio
  print("Loading audio from ", fname)
  fs.download(fname)
  fs_rate, audio = wavfile.read(fname)
  new_dtype = BITS_TO_DTYPE[8]
  if new_dtype != audio.dtype:
      current_range, new_range = DTYPE_RANGES[audio.dtype], DTYPE_RANGES[new_dtype]
      audio = ((audio - current_range[0]) / (current_range[1] - current_range[0]) * (new_range[1] - new_range[0]) + new_range[0]).astype(new_dtype)
  #Each sample is SPLIT length long, so we need to split into chunks of SPLIT * 2
  print("Done loading!")
  call(['rm', fname])
  return audio


class UpNet:
    def __init__(self, input_length, output_length, num_layers=4,
                 batch_size=128, learning_rate=1e-4, b1=0.99, b2=0.999):
        self.r = 2
        self.layers = num_layers


        n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
        n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]
        downsampling_l = []

        print('Generating model')

        # Downsampling layers
        self.conv_before = []
        for l, nf, fs in zip(range(num_layers), n_filters, n_filtersizes):
          conv = nn.Conv1D(1, nf, fs, stride=2)
          relu = nn.leaky_relu(0.2)(x)
          self.conv_before.append((conv, relu))
          print('D-Block: ', x.get_shape(), downsampling_l.append(x))

        # bottleneck layer
        with tf.name_scope('bottleneck_conv'):
            x = (Conv1D(n_filters[-1], n_filtersizes[-1], strides=2))(x)
            x = Dropout(0.5)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)

        # upsampling layers
        for l, nf, fs, l_in in reversed(list(zip(
                range(num_layers), n_filters, n_filtersizes, downsampling_l
        ))):
            with tf.name_scope('upsc_conv%d' % l):
                # (-1, n/2, 2f)
                x = (Conv1D(2 * nf, fs))(x)
                x = BatchNormalization()(x)
                x = Dropout(0.5)(x)
                x = Activation('relu')(x)
                
                # (-1, n, f)
                x = SubPixel1D(x, 2)
                x = merge.concatenate([x, l_in], axis=1) 
                # (-1, n, 2f)
                print('U-Block: ', x.get_shape())

        # final conv layer
        with tf.name_scope('lastconv'):
            x = Conv1D(2, 9)(x)
            x = SubPixel1D(x, 2)
            print(x.get_shape())
        

        self.predictions = Dense(output_length, input_shape=x.get_shape())(x)
        
        # self.output = x
        self.model = Model(inputs=self.input, outputs=self.predictions)


def train(epochs=1, model_name=None):
    
    # Initialize list of available data
    # input_directory = ROOTDIR + 'wav_{}/'.format(INPUT_SAMPLE_RATE)
    # output_directory = ROOTDIR + 'wav_{}/'.format(OUTPUT_SAMPLE_RATE)
    fs = Fileserver()

    input_directory_name = ROOTDIR + 'overfit_wav_input/'
    output_directory_name = ROOTDIR + 'overfit_wav_output/'
    
    input_dir = os.listdir(input_directory_name)[:20]
    output_dir = os.listdir(output_directory_name)[:20]
    fs.cd('overfit_wav_input')
    input_files = [load_raw_input(fn, fs) for fn in input_dir]
    fs.cd('../overfit_wav_output')
    output_files = [load_raw_input(fn, fs) for fn in output_dir]
    print(len(input_files[0]))
    # assert len(input_files) == len(output_files)
    # assert all([fn.endswith('.wav') for fn in input_files + output_files])
    pairs = list(zip(input_files, output_files))
    random.seed(0)
    random.shuffle(pairs)
    if model_name is None:
        # Initialize model
        upnet = UpNet(
            int(INPUT_SAMPLE_RATE * SAMPLE_LENGTH),
            int(OUTPUT_SAMPLE_RATE * SAMPLE_LENGTH),
        )
    else:
        # Load model from file
        upnet = None
    # Train
    upnet.model.compile(
        optimizer='adam', loss='mean_squared_error', metrics=['accuracy']
    )
    upnet.model.fit(input_files,output_files, epochs=1,verbose=2)


def eval():
    pass

train()