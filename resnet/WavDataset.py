from torch.utils.data import Dataset 
import torch
from scipy.io import wavfile
import os

ROOTDIR = '/home/abhishyant/bdisk/BRIANDISK/tensorpros/fma_small/'
class WavDataset(Dataset):

  def __init__(self, input_dirname, input_chunksize, output_dirname, output_chunksize):
    input_audio, output_audio = self.load_files(input_dirname, output_dirname)
    self.input_audio = self.chunk_audio(input_audio, input_chunksize)
    self.output_audio = self.chunk_audio(output_audio, output_chunksize)

  def load_raw_input(self, fname):
    # Reduce bitrate of audio
    fs_rate, audio = wavfile.read(fname)
    # new_dtype = BITS_TO_DTYPE[8]
    # if new_dtype != audio.dtype:
    #     current_range, new_range = DTYPE_RANGES[audio.dtype], DTYPE_RANGES[new_dtype]
    #     audio = ((audio - current_range[0]) / (current_range[1] - current_range[0]) * (new_range[1] - new_range[0]) + new_range[0]).astype(new_dtype)
    #Each sample is SPLIT length long, so we need to split into chunks of SPLIT * 2
    print("Done loading", fname)
    #call(['rm', fname])
    return audio
  
  def load_files(self, input_dirname, output_dirname):
    
    # Initialize list of available data
    input_directory = ROOTDIR + input_dirname
    output_directory = ROOTDIR + output_dirname
    input_dir = os.listdir(input_directory)
    output_dir = os.listdir(output_directory)
    #print("Loading FS")
    #fs = Fileserver()
    #print("Done Loading")

    
    #fs.cd('overfit_wav_input')
    #print(fs.ls())
    input_files = [self.load_raw_input(input_directory + fn) for fn in input_dir]
    #fs.cd('../overfit_wav_output')
    output_files = [self.load_raw_input(output_directory + fn) for fn in output_dir]
    

    return input_files, output_files
   


  
  def chunk_audio(self, audio, chunksize):
    audio_lim = (len(audio) // chunksize) * chunksize
    audio = audio[:audio_lim]
    audio_chunks = [torch.from_numpy(audio[i:i+chunksize]).float() for i in range(0,audio_lim, chunksize)]    
    return audio_chunks

  def __len__(self):
    return len(self.input_audio)
  
  def __getitem__(self, idx):
    return self.input_audio[idx], self.output_audio[idx]