# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Translate text or files using trained transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.transformer.utils import tokenizer
from official.utils.flags import core as flags_core
from official.transformer.utils.dataset import _load_records
from official.transformer.utils.dataset import _parse_example


_DECODE_BATCH_SIZE = 32
_EXTRA_DECODE_LENGTH = 100
_BEAM_SIZE = 4
_ALPHA = 0.6




def translate_file(
    estimator, input_file, output_file=None,
    print_all_translations=True):
  """Translate lines in file, and save to output file if specified.

  Args:
    estimator: tf.Estimator used to generate the translations.
    subtokenizer: Subtokenizer object for encoding and decoding source and
       translated lines.
    input_file: file containing lines to translate
    output_file: file that stores the generated translations.
    print_all_translations: If true, all translations are printed to stdout.

  Raises:
    ValueError: if output file is invalid.
  """


  def input_fn():
    """Created batched dataset of encoded inputs."""
    dataset = tf.data.Dataset.list_files(input_file, shuffle=False)
    dataset = dataset.apply(_load_records)
    dataset = dataset.map(_parse_example)
    return dataset.batch(batch_size=10)
    

  samples = []
  for i, prediction in enumerate(estimator.predict(input_fn)):
    output = prediction["outputs"]
    samples.extend(output)

  tf.logging.info("Writing to file")
  with tf.gfile.Open(output_file, "w") as f:
    for sample in samples:
      f.write("%s\n" % str(sample))




def main(unused_argv):
  from official.transformer import transformer_main


  if FLAGS.text is None and FLAGS.file is None:
    tf.logging.warn("Nothing to translate. Make sure to call this script using "
                    "flags --text or --file.")
    return


  # Set up estimator and params
  params = transformer_main.PARAMS_MAP[FLAGS.param_set]
  params["beam_size"] = _BEAM_SIZE
  params["alpha"] = _ALPHA
  params["extra_decode_length"] = _EXTRA_DECODE_LENGTH
  params["batch_size"] = _DECODE_BATCH_SIZE
  estimator = tf.estimator.Estimator(
      model_fn=transformer_main.model_fn, model_dir=FLAGS.model_dir,
      params=params)


  if FLAGS.file is not None:
    input_file = os.path.abspath(FLAGS.file)
    tf.logging.info("Translating file: %s" % input_file)
    if not tf.gfile.Exists(FLAGS.file):
      raise ValueError("File does not exist: %s" % input_file)

    output_file = None
    if FLAGS.file_out is not None:
      output_file = os.path.abspath(FLAGS.file_out)
      tf.logging.info("File output specified: %s" % output_file)

    translate_file(estimator, input_file, output_file)


def define_translate_flags():
  """Define flags used for translation script."""
  # Model flags
  flags.DEFINE_string(
      name="model_dir", short_name="md", default="../saved_models/",
      help=flags_core.help_wrap(
          "Directory containing Transformer model checkpoints."))
  flags.DEFINE_enum(
      name="param_set", short_name="mp", default="big",
      enum_values=["base", "big"],
      help=flags_core.help_wrap(
          "Parameter set to use when creating and training the model. The "
          "parameters define the input shape (batch size and max length), "
          "model configuration (size of embedding, # of hidden layers, etc.), "
          "and various other settings. The big parameter set increases the "
          "default batch size, embedding/hidden size, and filter size. For a "
          "complete list of parameters, please see model/model_params.py."))

  flags.DEFINE_string(
      name="text", default=None,
      help=flags_core.help_wrap(
          "Text to translate. Output will be printed to console."))
  flags.DEFINE_string(
      name="file", default=None,
      help=flags_core.help_wrap(
          "File containing text to translate. Translation will be printed to "
          "console and, if --file_out is provided, saved to an output file."))
  flags.DEFINE_string(
      name="file_out", default=None,
      help=flags_core.help_wrap(
          "If --file flag is specified, save translation to this file."))


if __name__ == "__main__":
  define_translate_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
