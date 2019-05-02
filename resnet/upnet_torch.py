import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader, Subset

from data import WavData

INPUT_SAMPLE_RATE = 8000
OUTPUT_SAMPLE_RATE = 44100
SAMPLE_LENGTH = 1

INPUT_LEN = INPUT_SAMPLE_RATE * SAMPLE_LENGTH
OUTPUT_LEN = OUTPUT_SAMPLE_RATE * SAMPLE_LENGTH

# ROOTDIR = '/home/abhishyant/bdisk/BRIANDISK/tensorpros/fma_small/'
ROOTDIR = '/Users/brianlevis/cs182/audio-upsampling/data'
# ROOTDIR = '/home/abhishyant/data/'

# Mount Brian's disk with curlftpfs :
#   Install curlftpfs with Homebrew
#   Make a directory to load the disk into such as mkdir ~/bdisk
#   Run sudo curlftpfs -o allow_other
#   cs182:donkeyballs@fileserver.brianlevis.com ~/bdisk
# Create validation set in bdisk:
#   Pick all filenames of the form 00*.wav in the wav_8000 folder and put it in
#   the val_input folder
#   Pick all filenames of the form 00*.wav in the wav_44100 folder and put it
#   in the val_output folder
#   Modify code so it splits up input and output songs into chunks of 1 second
#   and feeds the chunks into the network in batches, possibly using the
#   Dataloader object from Pytorch, need to look that up

#   Modify network so it adds in Leaky Relu and Batchnorm and Dropout after the
#   convolutional layers, as per the paper:
#   https://arxiv.org/pdf/1708.00853.pdf

DTYPE_RANGES = {
    np.dtype('float32'): (-1.0, 1.0), np.dtype('int32'): (-2147483648, 2147483647),
    np.dtype('int16'): (-32768, 32767), np.dtype('uint8'): (0, 255)
}
BITS_TO_DTYPE = {
    64: np.dtype('float32'), 32: np.dtype('int32'), 16: np.dtype('int16'), 8: np.dtype('uint8')
}

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# Network design modeled after
# https://github.com/kuleshov/audio-super-res


def SubPixel1D(I, r):
    b, w, r = I.size()
    X = I.permute(2, 1, 0)  # (r, w, b)
    X = X.reshape(1, r * w, b)  # (1, r*w, b)
    X = X.permute(2, 1, 0)
    return X


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

        # Downsampling layers
        self.conv_before = []
        for i, (l, nf, fs) in enumerate(zip(list(range(num_layers)), n_filters, n_filtersizes)):
            if i == 0:
                conv = nn.Conv1d(1, nf, fs, stride=2)
                bn = nn.BatchNorm1d(nf)
                do = nn.Dropout(p=0.1)
            else:
                conv = nn.Conv1d(n_filters[i - 1], nf, fs, stride=2)
                bn = nn.BatchNorm1d(nf)
                do = nn.Dropout(p=0.1)
            self.conv_before.append((conv, bn, do))

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
            if i == 0:
                conv = nn.Conv1d(n_filters[-1], 2 * nf, fs)
                bn = nn.BatchNorm1d(2 * nf)
                do = nn.Dropout(p=.1)
            else:
                conv = nn.Conv1d(n_filters[-i], 2 * nf, fs)
                bn = nn.BatchNorm1d(nf)
                do = nn.Dropout(p=.1)
            subpixel = nn.PixelShuffle(2)
            self.up_convs.append((conv, subpixel, bn, do))

        # final conv layer
        self.final_conv = nn.Conv1d(n_filters[0], 2, 9)
        self.output_fc = nn.Linear(self.fc_dimensions(n_filters, n_filtersizes), output_length)

        print(self.fc_dimensions(n_filters, n_filtersizes))

    def fc_dimensions(self, n_filters, n_filtersizes):
        def conv_dims(w, k, s, p=0):
            out = (w - k + 2 * p) / s + 1.0
            out = int(out)
            return out

        def subpixel_dims(shape, r):
            H = 1
            _, C, W = shape
            shape = [1, C / pow(r, 2), H * r, W * r]
            return shape[1:]

        # self.input_length = 8000
        shape = [1, 1, self.input_length]

        dl = []

        # down convs
        for i, (nf, fs) in enumerate(zip(n_filters, n_filtersizes)):
            _, _, w = shape
            cd = conv_dims(w=w, k=fs, s=2)
            dl.append(cd)
            shape = [1, nf, cd]

        # bottleneck with conv
        _, _, w = shape
        shape = [1, n_filters[-1], conv_dims(w=w, k=n_filtersizes[-1], s=2)]

        # upsample
        for i, (nf, fs, cd) in enumerate(reversed(list(zip(n_filters, n_filtersizes, dl)))):
            _, _, w = shape

            # up conv
            shape = [1, 2 * nf, conv_dims(w=w, k=fs, s=1)]

            # subpixel
            C, H, W = subpixel_dims(shape, 2)

            # view
            C, H, W = (1, C * H, W)

            # cat
            C, H, W = (C, H, W + cd)

            shape = [C, H, W]

        # final conv
        _, _, w = shape
        w = conv_dims(w=w, k=9, s=1)

        return w * 2

    def forward(self, x):

        x = x[:, :, :self.input_length]

        downsampling_l = [x]

        for (conv, bn, do) in self.conv_before:
            x = F.leaky_relu(conv(x).to(device)).to(device)
            x = do(bn(x).to(device)).to(device)
            downsampling_l.append(x)

        x = self.bottleneck(x).to(device)
        x = self.bottleneck_dropout(x).to(device)
        x = self.bottleneck_bn(x).to(device)

        for i, (conv, subpixel, bn, do) in enumerate(self.up_convs):
            x = do(bn(conv(x).to(device)).to(device)).to(device)
            x = x.unsqueeze(2)
            x = subpixel(x)[0]
            x = x.view(1, 2 * x.size()[0], x.size()[2])
            l_in = downsampling_l[len(downsampling_l) - 1 - i]
            x = torch.cat((x, l_in), -1)

        x = self.final_conv(x).to(device)
        x = SubPixel1D(x, 2)
        x = x.view(x.size()[0], x.size()[1])

        x = self.output_fc(x)

        return x


def load_model(model_name=None):
    if model_name is None:
        # Initialize model
        upnet = UpNet(
            int(INPUT_SAMPLE_RATE * SAMPLE_LENGTH),
            int(OUTPUT_SAMPLE_RATE * SAMPLE_LENGTH),
        ).to(device)
    else:
        # Load model from file
        upnet = torch.load(model_name).to(device)
        # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(upnet.parameters(), lr=1e-3, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    return upnet, criterion, optimizer, scheduler


def train(model_data, data, val_data, num_epochs=1000):
    model, criterion, optimizer, scheduler = model_data
    i = 0
    for epoch in range(num_epochs):
        # Training
        print('epoch {}'.format(epoch))
        for i, (input_file, output_file) in enumerate(data):

            # Transfer to GPU
            input_file = input_file.to(device)
            output_file = output_file.to(device)

            model.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = model.forward(input_file)
            print("Loss")
            loss = criterion(outputs, output_file)
            # Backward and optimize
            print("Backward")
            loss.backward()
            print("Step")
            optimizer.step()

            if (i) % 100 == 0:
                val_input, val_output = val_data.__iter__().__next__()
                val_input.to(device)
                val_output.to(device)
                model.eval()
                outputs = model.forward(val_input)
                val_loss = criterion(outputs, val_output)
                with open('train_log.txt', 'w') as f:
                    f.write('{}, {}'.format(i, loss.item()))
                with open('val_log.txt', 'w') as f:
                    f.write('{}, {}').format(i, val_loss.item())
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, len(data), loss.item(), val_loss.item()))
            i = i + 1

        scheduler.step()
        subprocess.call(['rm', 'model.ckpt'])
        torch.save(model.state_dict(), 'model.ckpt')


# train_input_dir = 'wav_{}/'.format(INPUT_SAMPLE_RATE)
# train_output_dir = 'wav_{}/'.format(OUTPUT_SAMPLE_RATE)
# train_dataset = WavDataset(train_input_dir, INPUT_LEN, train_output_dir, OUTPUT_LEN)
# train_dl = Dataloader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_input_dir = 'val_input'
# val_output_dir = 'val_output'
# val_dataset = WavDataset(val_input_dir, INPUT_LEN, val_output_dir, OUTPUT_LEN)
# val_dl = Dataloader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
#
# model_data = load_model()
# train(model_data, train_dl, val_dl)

if __name__ == "__main__":
    dataset = WavData(INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, SAMPLE_LENGTH, ROOTDIR)

    dataset_len = len(dataset)
    train_len = int(dataset_len * 0.8)
    eval_len = dataset_len - train_len

    train_dataset = Subset(dataset, list(range(train_len)))
    eval_dataset = Subset(dataset, list(range(train_len, eval_len + train_len)))

    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dl = DataLoader(eval_dataset, batch_size=32, shuffle=True, num_workers=4)

    model_data = load_model()
    train(model_data, train_dl, val_dl)
