import os

import scipy.io.wavfile
import torch.utils.data


class WavData(torch.utils.data.Dataset):
    def __init__(self, input_rate, output_rate, sample_length, path):
        """Initialize interface to the 8-bit wav dataset.

        :param input_rate: Input sample rate
        :param output_rate: Output sample rate
        :param sample_length: Length of model input in seconds
        :param path: Path to wav_<Hz>/ folders
        """
        # TODO: TREAT *SAMPLES* AS ITEMS - NOT WAVS, SO AS TO PROPERLY
        #       USE TORCH'S DATA UTILS
        #       I am still working on this
        assert (sample_length * input_rate).is_integer()
        assert (sample_length * output_rate).is_integer()

        self.input_rate = input_rate
        self.output_rate = output_rate
        self.input_sample_size = int(sample_length * input_rate)
        self.output_sample_size = int(sample_length * output_rate)

        self.path = path if path.endswith('/') else path + '/'
        self.input_path = '{}wav_{}/'.format(self.path, self.input_rate)
        self.output_path = '{}wav_{}/'.format(self.path, self.output_rate)
        self.filenames = sorted(set([
            fn for fn in os.listdir(self.input_path) if fn.endswith('.wav')
        ]).intersection([
            fn for fn in os.listdir(self.output_path) if fn.endswith('.wav')
        ]))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        # Load files
        _, input_wav = scipy.io.wavfile.read(self.input_path + self.input_rate)
        _, output_wav = scipy.io.wavfile.read(self.output_path + self.output_rate)
        # Truncate data
        num_samples = len(input_wav) // self.input_sample_size
        input_wav = input_wav[:num_samples * self.input_sample_size].reshape((num_samples, -1))
        output_wav = output_wav[:num_samples * self.output_sample_size].reshape((num_samples, -1))
        return input_wav, output_wav

    @staticmethod
    def _load_data_annotation(self):
        pass