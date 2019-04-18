import tensorflow as tf 
from scipy.io import wavfile
from scipy import signal
import six
import numpy as np
import librosa

DTYPE_RANGES = {
    np.float32: (-1.0, 1.0), np.int32: (-2147483648, 2147483647), 
    np.dtype('int16'): (-32768, 32767), np.dtype('uint8'): (0, 255)
}
BITS_TO_DTYPE = {
    64: np.float32, 32: np.int32, 16: np.dtype('int16'), 8: np.dtype('uint8')
}




def load_raw_input(fname):
  # Reduce bitrate of audio
  fs, audio = wavfile.read(fname)
  audio = signal.resample(audio, int(len(audio) / 441))
  audio = audio.astype(np.float32)
  new_dtype = BITS_TO_DTYPE[8]
  if new_dtype != audio.dtype:
      audio = (
          (audio + 1.0) / (1.0 + 1.0) *
          (255) 
      ).astype(new_dtype)
  return audio

def create_input_target(audio,i):
  print("Resampling...")
  audio = audio[i:i+100]
  target_audio = [audio[i] for i in range(len(audio)) if i % 2 == 1]
  audio = [audio[i] for i in range(len(audio)) if i % 2 == 0]
  return audio, target_audio

def dict_to_example(dictionary):
  """Converts a dictionary of string->int to a tf.Example."""
  features = {}
  for k, v in six.iteritems(dictionary):
    features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
  return tf.train.Example(features=tf.train.Features(feature=features))
fname = "/Users/abhishyant/Music/distorted/jlucas.wav"
writer = tf.python_io.TFRecordWriter('./train/jlucas'  + '.tfrecord')
audio = load_raw_input(fname)
for i in range(0, len(audio),100):
  
  input_audio, target_audio = create_input_target(audio,i)
  audio_dict = {
    "inputs": input_audio,
    "targets": target_audio
  }
  example = dict_to_example(audio_dict)
  writer.write(example.SerializeToString())
writer.close()
