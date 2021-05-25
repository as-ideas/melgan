import os
import glob
from pathlib import Path

import tqdm
import torch
import argparse
from scipy.io.wavfile import write
import numpy as np
from model.generator import Generator
from utils.hparams import HParam, load_hparam_str

MAX_WAV_VALUE = 32768.0


def main(args):
    model_params = {
        'nvidia_tacotron2_LJ11_epoch6400': {
            'mel_channel': 80,
            'model_url': 'https://github.com/seungwonpark/melgan/releases/download/v0.3-alpha/nvidia_tacotron2_LJ11_epoch6400.pt',
        },
    }
    params = model_params['nvidia_tacotron2_LJ11_epoch6400']
    model = Generator(params['mel_channel'])
    checkpoint = torch.hub.load_state_dict_from_url(params['model_url'], progress=True, map_location="cpu")
    model.load_state_dict(checkpoint['model_g'])
    model.to(torch.device('cpu'))
    model.eval(inference=True)
    print(model)
    input = glob.glob(os.path.join(args.input_folder, '*.npy'))

    with torch.no_grad():
        print(input)
        for melpath in tqdm.tqdm(input):
            #mel = torch.load(melpath)
            mel = torch.tensor(np.load(melpath))
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel

            audio = model.gta(mel)
            audio = audio.cpu().detach()
            file = Path(melpath).stem
            out_path = f'/tmp/gta/{file}_gta.mel'
            torch.save(audio, out_path)
            #np.save(out_path, audio, allow_pickle=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help="directory of mel-spectrograms to invert into raw audio. ")
    args = parser.parse_args()

    main(args)
