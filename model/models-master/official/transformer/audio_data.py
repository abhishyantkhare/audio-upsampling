import tensorflow as tf 
import scipy.io.wavfile
from scipy import signal
import numpy as np 
import os

rootdir = "/users/abhishyant/Music/distorted/"

def load_raw_input(fname):
  fs, audio = scipy.io.wavfile.read(fname, mmap=False)

  # Convert to mono if necessary
  mono_audio = audio
  if len(audio.shape) > 1:
      mono_audio = audio.mean(axis=1)
  audio = mono_audio.astype(audio.dtype)

  return audio

def create_input_target(audio):
  print("Resampling...")
  target_audio = [0] + audio[1:]
  return audio, target_audio

def parse_audio(fname):
  audio = load_raw_input(fname)
  return create_input_target(audio)




def gen_dataset(shuffle=False,batch_size=10):
  features = []
  labels = []
  for path, dirs, files in os.walk(rootdir):
    for filename in files:
        if filename[-3:] == 'wav':
          fullpath = os.path.join(path, filename)
          input_audio, target_audio = parse_audio(fullpath)
          features.append(input_audio)
          labels.append(target_audio)
  
  print("Generating Dataset...")

  dataset = tf.data.Dataset.from_generator(lambda: (features, labels), output_types=(tf.int32, tf.int32), output_shapes= (tf.TensorShape([None]), tf.TensorShape([None])))
  dataset = dataset.batch(batch_size)
  print("Done")
  return dataset

def audio_input_fn(params):
  return gen_dataset(batch_size=1)


