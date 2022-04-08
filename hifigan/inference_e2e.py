from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
import librosa
import pickle
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    # filelist = os.listdir(a.input_mels_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()

    data = pickle.load(open(a.input_mels_dir, 'rb'))
    with torch.no_grad():
        # for i, filname in enumerate(filelist):
        for _, x in enumerate(data):
            # x = np.load(os.path.join(a.input_mels_dir, filname))
            filename = x[0]
            x = x[1]
            x = torch.FloatTensor(x).to(device)
            x = torch.unsqueeze(x, dim=-1)
            x = x.permute(2, 1, 0)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            # audio = audio * MAX_WAV_VALUE
            audio = audio.view(-1).cpu().numpy().astype('float')
            # audio = audio.cpu().numpy()

            output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '_generated_e2e.wav')
            # write(output_file, h.sampling_rate, audio)
            write(output_file, 16000, audio)
            # librosa.output.write_wav(output_file, audio, sr=16000)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='/home/hongcz/alab/code/VF-VC/results.pkl')
    parser.add_argument('--output_dir', default='generated_files_from_mel')
    parser.add_argument('--checkpoint_file', default='/home/hongcz/alab/code/VF-VC/checkpoint/hifigan/g_03280000')
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

