import os
import random

import torch
import torch.nn as nn

import torch.nn.functional as F

from scipy.io import wavfile
import numpy as np
from fileserver import Fileserver
from subprocess import call

INPUT_SAMPLE_RATE = 8000
OUTPUT_SAMPLE_RATE = 44100
SAMPLE_LENGTH = 1
ROOTDIR = '/Users/abhishyant/bdisk/BRIANDISK/tensorpros/fma_small/'

#TODO:
# Create validation set
# Load in data from Brian's disk by mounting it with curlftpfs and reading in filenames with os.listdir(), not the Fileserver object
# Modify code so it splits up input and output songs into chunks of 1 second each and feeds that into the network, possibly using 
# the Dataloader object from Pytorch
# Modify network so it adds in Leaky Relu and Batchnorm and Dropout after the convolutional layers, as per the paper: https://arxiv.org/pdf/1708.00853.pdf
# Train and see!


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
    X = I.permute(2, 1, 0)  # (r, w, b)
    X = X.reshape(1, r*w, b)  # (1, r*w, b)
    X = X.permute(2, 1, 0)
    return X

def load_raw_input(fname):
  # Reduce bitrate of audio
  print("Loading audio from ", fname)
  fname = fname.split("/")[1]
  print(fname)
  #fs.download(fname)
  fs_rate, audio = wavfile.read(fname)
  new_dtype = BITS_TO_DTYPE[8]
  if new_dtype != audio.dtype:
      current_range, new_range = DTYPE_RANGES[audio.dtype], DTYPE_RANGES[new_dtype]
      audio = ((audio - current_range[0]) / (current_range[1] - current_range[0]) * (new_range[1] - new_range[0]) + new_range[0]).astype(new_dtype)
  #Each sample is SPLIT length long, so we need to split into chunks of SPLIT * 2
  print("Done loading!")
  #call(['rm', fname])
  return audio


class UpNet(nn.Module):
    def __init__(self, input_length, output_length, num_layers=4,
                 batch_size=128, learning_rate=1e-4, b1=0.99, b2=0.999):
        super(UpNet, self).__init__()
        self.r = 2
        self.layers = num_layers
        self.input_length = input_length
        self.output_length = output_length

        n_filters = [128, 256, 512, 512]
        n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]

        print('Generating model')

        # Downsampling layers
        self.conv_before = []
        for i, (l, nf, fs) in enumerate(zip(list(range(num_layers)), n_filters, n_filtersizes)):
          if i == 0:
            conv = nn.Conv1d(1, nf, fs, stride=2)
          else:
            conv = nn.Conv1d(n_filters[i-1], nf, fs, stride=2)
          self.conv_before.append(conv)

        # bottleneck layer
        self.bottleneck = nn.Conv1d(n_filters[-1], n_filters[-1], n_filtersizes[-1], stride=2)
        self.bottleneck_dropout = nn.Dropout(p=0.5)
        self.bottleneck_bn = nn.BatchNorm1d(n_filters[-1])
        # x = LeakyReLU(0.2)(x)

        # upsampling layers
        self.up_convs = []
        for i, (l, nf, fs) in enumerate(reversed(list(zip(
                range(num_layers), n_filters, n_filtersizes
        )))):
              # (-1, n/2, 2f)
            if i == 0:
              conv = nn.Conv1d(n_filters[-1], 2 * nf, fs)
            else:
              conv = nn.Conv1d(n_filters[-i], 2*nf, fs)
            subpixel = nn.PixelShuffle(2)
            self.up_convs.append((conv,subpixel))
            
            # x = BatchNormalization()(x)
            # x = Dropout(0.5)(x)
            # x = Activation('relu')(x)
            
            # (-1, n, f)
            # subpix = SubPixel1D(x, 2)
            # x = merge.concatenate([x, l_in], axis=1) 
            # (-1, n, 2f)

        # final conv layer
        self.final_conv = nn.Conv1d(n_filters[0], 2, 9)
        self.output_fc = nn.Linear(2968, output_length)

          # x = SubPixel1D(x, 2)
          # print(x.size())


        # self.predictions = Dense(output_length, input_shape=x.get_shape())(x)
        
        # self.output = x
        # self.model = Model(inputs=self.input, outputs=self.predictions)

    def forward(self, x):
      x = x[:,:,:self.input_length]
      downsampling_l = [x]
      for conv in self.conv_before:
        x = F.leaky_relu(conv(x))
        downsampling_l.append(x)
      x = self.bottleneck(x)
      x = self.bottleneck_dropout(x)
      x = self.bottleneck_bn(x)
      for i, (conv, subpixel) in enumerate(self.up_convs):
        x = conv(x)
        x = x.unsqueeze(2)
        x = subpixel(x)[0]
        x = x.view(1, 2*x.size()[0], x.size()[2])
        l_in = downsampling_l[len(downsampling_l) - 1 - i]
        x = torch.cat((x, l_in), -1)
      x = self.final_conv(x)
      x = SubPixel1D(x, 2)
      x = x.view(x.size()[0], x.size()[1])
      x = self.output_fc(x)
      return x 


def load_files():
    
  # Initialize list of available data
  input_directory = ROOTDIR + 'wav_{}/'.format(INPUT_SAMPLE_RATE)
  output_directory = ROOTDIR + 'wav_{}/'.format(OUTPUT_SAMPLE_RATE)
  input_dir = os.listdir(input_directory)
  output_dir = os.listdir(output_directory)
  #print("Loading FS")
  #fs = Fileserver()
  #print("Done Loading")

  
  #fs.cd('overfit_wav_input')
  #print(fs.ls())
  input_files = [load_raw_input(input_directory + fn) for fn in input_dir]
  #fs.cd('../overfit_wav_output')
  output_files = [load_raw_input(output_directory + fn) for fn in output_dir]
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

def train(model_data, data, num_epochs = 1000):
  input_files, output_files = data
  model, criterion, optimizer, scheduler = model_data
  i = 0
  for epoch in range(num_epochs):
    # Training
    print('epoch {}'.format(epoch))
    for input_file, output_file in zip(input_files, output_files):
        # Transfer to GPU
        model.train()
        optimizer.zero_grad()

        # Forward pass
        input_file = torch.from_numpy(input_file).float()
        input_file = input_file.view(1, 1, input_file.size()[0])
        output_file = torch.from_numpy(output_file).float()
        output_file = output_file[:OUTPUT_SAMPLE_RATE*SAMPLE_LENGTH]
        outputs = model.forward(input_file)
        loss = criterion(outputs, output_file)
        # Backward and optimize
        loss.backward()
        optimizer.step()

        if (i) % 10 == 0:
            model.eval()
            print ('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, len(input_files), loss.item()))
        i = i + 1
           
    scheduler.step()
   


def eval():
    pass

data = load_files()
model_data = load_model()
train(model_data, data)
