import os
import glob
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
    #checkpoint = torch.load(args.checkpoint_path)
    audios = []
    input = glob.glob(os.path.join(args.input_folder, '*.mel'))

    with torch.no_grad():
        print(input)
        for melpath in tqdm.tqdm(input):
            mel = torch.load(melpath)
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel

            audio = model.gen_from_gta(mel)
            audio = audio.cpu().detach().numpy()
            audios.append(audio)
            audios.append(np.zeros(10000, dtype=np.int16))
            out_path = melpath.replace('.mel', '_reconstructed_epoch%04d.wav' % checkpoint['epoch'])
            write(out_path, 22050, audio)

    article = np.concatenate(audios[:], axis=0)
    write('/tmp/article_melgan.wav', 22050, article)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help="directory of mel-spectrograms to invert into raw audio. ")
    args = parser.parse_args()

    main(args)
