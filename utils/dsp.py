import math
import torch
import numpy as np
import librosa
from scipy.signal import lfilter

class hp:
    n_fft = 2048
    num_mels = 80
    sample_rate = 22050
    fmin = 40
    min_level_db = -100
    ref_level_db = 20
    hop_length = 256
        win_length = 1024


def load_wav(path):
    return librosa.load(path, sr=hp.sample_rate)[0]


def save_wav(x, path):
    librosa.output.write_wav(path, x.astype(np.float32), sr=hp.sample_rate)


def linear_to_mel(spectrogram):
    return librosa.feature.melspectrogram(
        S=spectrogram, sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin)

'''
def build_mel_basis():
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin)
'''

def normalize(S):
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)


def denormalize(S):
    return (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def spectrogram(y):
    D = stft(y)
    S = amp_to_db(np.abs(D)) - hp.ref_level_db
    return normalize(S)


def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)

def raw_melspec(y):
    D = stft(y)
    S = linear_to_mel(np.abs(D))
    return S

def stft(y):
    return librosa.stft(
        y=y,
        n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
