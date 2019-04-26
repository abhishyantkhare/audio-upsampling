import tensorflow as tf 
from scipy.io import wavfile
from scipy import signal
import six
import numpy as np
import librosa
import os
from data_utils.fileserver import Fileserver
from subprocess import call


DTYPE_RANGES = {
    np.float32: (-1.0, 1.0), np.int32: (-2147483648, 2147483647), 
    np.dtype('int16'): (-32768, 32767), np.dtype('uint8'): (0, 255)
}
BITS_TO_DTYPE = {
    64: np.float32, 32: np.int32, 16: np.dtype('int16'), 8: np.dtype('uint8')
}

SPLIT = 256


def load_raw_input(fname):
  # Reduce bitrate of audio
  print("Loading audio...")
  fs.download(fname)
  fs_rate, audio = wavfile.read(fname)
  new_dtype = BITS_TO_DTYPE[8]
  if new_dtype != audio.dtype:
      current_range, new_range = DTYPE_RANGES[audio.dtype], DTYPE_RANGES[new_dtype]
      audio = ((audio - current_range[0]) / (current_range[1] - current_range[0]) * (new_range[1] - new_range[0]) + new_range[0]).astype(new_dtype)
  #Each sample is SPLIT length long, so we need to split into chunks of SPLIT * 2
  end_lim = (len(audio) // (SPLIT * 2)) * (SPLIT * 2)
  call(['rm', fname])
  return audio[:end_lim]

def create_input_target(audio):
  target_audio = [audio[i] for i in range(len(audio)) if i % 2 == 1]
  input_audio = [audio[i] for i in range(len(audio)) if i % 2 == 0]
  return input_audio, target_audio

def write_records(audio, file, path, fs):
  print("Writing TFRecord..")
  tf_name = file.split(".")[0] + ".tfrecord"
  tf_path_local = os.path.join('./', tf_name)
  writer = tf.python_io.TFRecordWriter(tf_path_local)
  for i in range(0,len(audio), SPLIT*2):
    input_audio, target_audio = create_input_target(audio[i:i+SPLIT*2])
    audio_dict = {
    "inputs": input_audio,
    "targets": target_audio
    }
    example = dict_to_example(audio_dict)
    writer.write(example.SerializeToString())
  writer.close() 
  fs.upload(tf_path_local)
  call(['rm', './' + tf_path_local])

def dict_to_example(dictionary):
  """Converts a dictionary of string->int to a tf.Example."""
  features = {}
  for k, v in six.iteritems(dictionary):
    features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
  return tf.train.Example(features=tf.train.Features(feature=features))

def process_files(rootdir,fs):
  for fname in fs.ls():
    print("Processing ", fname)
    fname = fname.split('/')[1]
    audio = load_raw_input(fname)
    write_records(audio, fname , rootdir,fs)
    print("Done!")



root_dirs = [4000,8000,11025, 22050,44100]
for dirnum in root_dirs:
  ROOTDIR = "./wav_{}".format(dirnum)
  fs = Fileserver()
  fs.cd(ROOTDIR)
  process_files(ROOTDIR,fs)

