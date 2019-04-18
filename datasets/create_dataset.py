import os

import librosa
import numpy as np
from data_utils import Fileserver, downsample_audio_file


CORRUPT_FILES = {'mp3/099134.mp3', 'mp3/108925.mp3'}


def convert_mp3s(fs):
    mp3_filenames = [
        fn for fn in fs.ls('mp3') if fn.endswith('.mp3') and fn not in CORRUPT_FILES
    ]
    existing_files = set(fs.ls('wav'))
    for sample_rate in [44100, 22050, 11025, 8000, 4000, 2000, 1000]:
        dir_name = 'wav_{}'.format(sample_rate)
        existing_files.update(fs.ls(dir_name))
    for f_in in mp3_filenames:
        # print(f_in)
        # Create 44.1k, 16-bit stereo file for librosa/resampy downsampling
        f_out = f_in.replace('mp3', 'wav')
        if f_out not in existing_files:
            fs.download(f_in)
            local_f_in, local_f_out = fs.get_local_file(f_in), fs.get_local_file(f_out)
            os.system('ffmpeg -i {} {}'.format(local_f_in, local_f_out))
            fs.upload(f_out)
            download_new = False
        else:
            download_new = True
        # Create filtered files
        f_in = f_out
        local_f_in = fs.get_local_file(f_in)
        for sample_rate in [44100, 22050, 11025, 8000, 4000, 2000, 1000]:
            dir_name = 'wav_{}'.format(sample_rate)
            f_out = f_in.replace('wav/', dir_name + '/')
            local_f_out = fs.get_local_file(f_out)
            if f_out in existing_files:
                continue
            if download_new:
                fs.download(f_in)
                download_new = False
            downsample_audio_file(local_f_in, local_f_out, sample_rate, 8)
            fs.upload(f_out)
        fs.clear_cache()


fs = Fileserver()
convert_mp3s(fs)
del fs
