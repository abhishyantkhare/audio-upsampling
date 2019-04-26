import os
import random

import torch
import torch.nn as nn

import torch.nn.functional as F

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
        for i, l, nf, fs in enumerate(zip(range(num_layers), n_filters, n_filtersizes)):
          if i == 0:
            conv = nn.Conv1D(1, nf, fs, stride=2)
          else:
            conv = nn.Conv1D(n_filters[i-1], nf, fs, stride=2)
          self.conv_before.append(conv)

        # bottleneck layer
        self.bottleneck = nn.Conv1D(nfilters[-1], n_filters[-1], n_filtersizes[-1], strides=2)
        self.bottleneck_dropout = nn.Dropout(p=0.5)
        self.bottleneck_bn = nn.BatchNormalization()
        # x = LeakyReLU(0.2)(x)

        # upsampling layers
        self.up_convs = []
        for i, l, nf, fs, l_in in enumerate(reversed(list(zip(
                range(num_layers), n_filters, n_filtersizes, downsampling_l
        )))):
              # (-1, n/2, 2f)
            if i == 0:
              conv = nn.Conv1D(n_filters[-1], 2 * nf, fs)
            else:
              conv = nn.Conv1D(n_filters[-i], 2*nf, fs)
            self.up_convs.append((conv, l_in))
            
            # x = BatchNormalization()(x)
            # x = Dropout(0.5)(x)
            # x = Activation('relu')(x)
            
            # (-1, n, f)
            # subpix = SubPixel1D(x, 2)
            # x = merge.concatenate([x, l_in], axis=1) 
            # (-1, n, 2f)

        # final conv layer
        self.final_conv = Conv1D(n_filters[0], 2, 9)
        self.output_fc = nn.Linear(x.size()[1], output_length)

          # x = SubPixel1D(x, 2)
          # print(x.size())


        # self.predictions = Dense(output_length, input_shape=x.get_shape())(x)
        
        # self.output = x
        # self.model = Model(inputs=self.input, outputs=self.predictions)

    def forward(self, x):
      for conv in self.conv_before:
        x = conv(x)
      x = self.bottleneck(x)
      x = self.bottleneck_dropout(x)
      x = self.bottleneck_bn(x)
      for conv, l_in in self.up_convs:
        x = conv(x)
        x = SubPixel1D(x, 2)
        x = torch.cat((x, l_in), 1)
      x = self.final_conv(x)
      x = self.output_fc(x)
      return x 


def load_files():
    
  # Initialize list of available data
  # input_directory = ROOTDIR + 'wav_{}/'.format(INPUT_SAMPLE_RATE)
  # output_directory = ROOTDIR + 'wav_{}/'.format(OUTPUT_SAMPLE_RATE)
  print("Loading FS")
  fs = Fileserver()
  print("Done Loading")

  
  fs.cd('overfit_wav_input')
  print(fs.ls())
  input_files = [load_raw_input(fn, fs) for fn in fs.ls()]
  fs.cd('../overfit_wav_output')
  output_files = [load_raw_input(fn, fs) for fn in fs.ls()]
  print(len(input_files[0]))
  # assert len(input_files) == len(output_files)
  # assert all([fn.endswith('.wav') for fn in input_files + output_files])
  pairs = list(zip(input_files, output_files))
  random.seed(0)
  random.shuffle(pairs)
  return input_files, output_files
   
    # Train

def load_model(model_name=None):
  if model_name is None:
    # Initialize model
    upnet = UpNet(
        int(INPUT_SAMPLE_RATE * SAMPLE_LENGTH),
        int(OUTPUT_SAMPLE_RATE * SAMPLE_LENGTH),
    )
  else:
        # Load model from file
    upnet = None
    # Loss and optimizer
  criterion =  nn.MSELoss()
  optimizer = torch.optim.Adam(upnet.parameters(), lr=1e-3,weight_decay=.01)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
  return upnet, criterion, optimizer, scheduler

def train(model_data, data, num_epochs = 1):
  model, criterion, optimizer, scheduler = model_data
  for epoch in range(num_epochs):
    # Training
    print('epoch {}'.format(epoch))

    for i, input_file, output_file in enumerate(data):
        # Transfer to GPU
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model.forward(input_file)
        loss = criterion(outputs, output_file)
        # Backward and optimize
        loss.backward()
        optimizer.step()

        if (i) % 25 == 0:
            model.eval()
            print ('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, len(data), loss.item()))
           
    scheduler.step()
   


def eval():
    pass

data = load_files()
model_data = load_model()
train(model_data, data)