import contextlib
import json
import os
import time
import wave

import scipy.io.wavfile
import torch.utils.data


ANNOTATION_FILE = 'file_sizes.json'


class WavData(torch.utils.data.Dataset):
    def __init__(self, input_rate, output_rate, sample_length, path):
        """Initialize interface to the 8-bit wav dataset.

        :param input_rate: Input sample rate
        :param output_rate: Output sample rate
        :param sample_length: Length of model input in seconds
        :param path: Path to wav_<Hz>/ folders
        """
        t = time.time()
        print("Initializing loader")
        sample_length = float(sample_length)
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

        # Compute number of samples in each file
        self.file_sizes = {}
        file_index_path = self.path + ANNOTATION_FILE
        if os.path.exists(file_index_path):
            with open(file_index_path, 'r') as fp:
                self.file_sizes = json.load(fp)
        for r, p in [
            (self.input_rate, self.input_path),
            (self.output_rate, self.output_path)
        ]:
            r = str(r)
            min_length = float('inf')
            if r not in self.file_sizes:
                print("Indexing", r)
                for fn in self.filenames:
                    with contextlib.closing(wave.open(p + fn, 'r')) as fp:
                        min_length = min(fp.getnframes(), min_length)
                assert type(min_length) == int
                self.file_sizes[r] = min_length
                with open(file_index_path, 'w') as fp:
                    json.dump(self.file_sizes, fp)

        self.samples_per_file = min(
            self.file_sizes[str(self.input_rate)] // self.input_sample_size,
            self.file_sizes[str(self.output_rate)] // self.output_sample_size
        )

        self.cache = [None, None, None]
        print("Loader initialized", time.time() - t)

    def __len__(self):
        return len(self.filenames) * self.samples_per_file

    def __getitem__(self, index):
        filename = self.filenames[index]
        if self.cache[0] != filename:
            self.cache[0] = filename
            _, input_wav = scipy.io.wavfile.read(self.input_path + filename)
            _, output_wav = scipy.io.wavfile.read(self.output_path + filename)
            self.cache[1] = input_wav[:self.samples_per_file * self.input_sample_size].reshape((self.samples_per_file, -1))
            self.cache[2] = output_wav[:self.samples_per_file * self.output_sample_size].reshape((self.samples_per_file, -1))
        return self.cache[1][index], self.cache[2][index]


if __name__ == "__main__":
    dataset = WavData(8000, 44100, 0.5, '/Users/brianlevis/cs182/audio-upsampling/data')
