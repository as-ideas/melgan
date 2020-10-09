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
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])

    model = Generator(hp.audio.n_mel_channels)
    model.load_state_dict(checkpoint['model_g'])
    model.eval(inference=False)

    audios = []
    input = glob.glob(os.path.join(args.input_folder, '*.mel'))
    input.sort()
    print(input)
    #input.sort(key=lambda x: int(x.split('/')[-1].split('_')[0]))
    #input = input[2:]

    with torch.no_grad():
        print(input)
        for melpath in tqdm.tqdm(input):
            mel = torch.load(melpath)
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel

            audio = model.inference(mel)
            audio = audio.cpu().detach().numpy()
            audios.append(audio)
            audios.append(np.zeros(10000, dtype=np.int16))
            out_path = melpath.replace('.mel', '_reconstructed_epoch%04d.wav' % checkpoint['epoch'])
            write(out_path, hp.audio.sampling_rate, audio)

    article = np.concatenate(audios, axis=0)
    write('/tmp/article_melgan.wav', hp.audio.sampling_rate, article)


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
