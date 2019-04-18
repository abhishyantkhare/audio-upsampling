import librosa
import numpy as np
from scipy.io import wavfile

DTYPE_RANGES = {
    np.dtype('float32'): (-1.0, 1.0), np.dtype('int32'): (-2147483648, 2147483647),
    np.dtype('int16'): (-32768, 32767), np.dtype('uint8'): (0, 255)
}
BITS_TO_DTYPE = {
    64: np.dtype('float32'), 32: np.dtype('int32'), 16: np.dtype('int16'), 8: np.dtype('uint8')
}


# def downsample_audio(audio, sample_rate, bits_per_sample, filter):
#     # Filter audio
#     if filter:
#         print('Filtering')
#         bass_freq = 250
#         fc = 500 / (sample_rate / 2)
#         b, a = signal.cheby1(8, 5, fc, btype='highpass', output='ba')
#         audio = signal.filtfilt(b, a, audio, axis=0).astype(audio.dtype)
#     # # Resample
#     # print('resampling')
#     # if sample_rate != fs:
#     #     resampled_audio = signal.resample(filtered_audio, int(len(audio) * sample_rate / fs))
#     # else:
#     #     resampled_audio = filtered_audio
#     # Set bitrate via dtype
#     new_dtype = BITS_TO_DTYPE[bits_per_sample]
#     if new_dtype != audio.dtype:
#         current_range, new_range = DTYPE_RANGES[audio.dtype], DTYPE_RANGES[new_dtype]
#         audio = ((audio - current_range[0]) / (current_range[1] - current_range[0]) * (new_range[1] - new_range[0]) + new_range[0])
#     return audio.astype(new_dtype)


def downsample_audio_file(input_filename, output_filename, sample_rate, bits_per_sample):
    # Downsample and save (librosa supports only 16-bits)
    y, sr = librosa.core.load(input_filename, sr=sample_rate, mono=True)
    librosa.output.write_wav(output_filename, y, sr)
    # Reduce bitrate of audio
    fs, audio = wavfile.read(output_filename)
    new_dtype = BITS_TO_DTYPE[bits_per_sample]
    if new_dtype != audio.dtype:
        current_range, new_range = DTYPE_RANGES[audio.dtype], DTYPE_RANGES[new_dtype]
        audio = (
            (audio - current_range[0]) / (current_range[1] - current_range[0]) *
            (new_range[1] - new_range[0]) + new_range[0]
        ).astype(new_dtype)
    wavfile.write(output_filename, rate=fs,  data=audio)
