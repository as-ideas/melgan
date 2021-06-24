from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from pathlib import Path

if __name__ == '__main__':

    mel_dir = '/Users/cschaefe/workspace/ForwardTacotron/data/mel'
    mel_files = Path(mel_dir).glob('**/*.mel')
    mel_files = list(mel_files)
    mels = [torch.load(f).numpy() for f in mel_files]
    mels = np.concatenate(mels, axis=1)
    mean, std = np.mean(mels), np.std(mels)
    sil = (-11.51 - mean) / std
    print(f'mean {mean} std {std} sil {sil}')
