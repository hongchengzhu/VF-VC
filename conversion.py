import os
import pickle
import torch
import glob
import json
import argparse
from hifigan.env import AttrDict
import numpy as np
from math import ceil
import librosa
from synthesis import build_model
from synthesis import wavegen
from model_vc import Generator
from hifigan.models import Generator as hifi_Generator
from scipy.io.wavfile import write

# emerge autovc with hifigan

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad


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


def inference(args):
    # device = 'cuda:0'
    G = Generator(32, 256, 512, 32).eval().to(device)

    g_checkpoint = torch.load(args.autovc_checkpoint, map_location={'cuda:1': 'cuda:0'})
    G.load_state_dict(g_checkpoint['model'])

    metadata = pickle.load(open(args.spec, "rb"))

    spect_vc = []

    for sbmt_i in metadata:
        x_org = sbmt_i[2]
        x_org, len_pad = pad_seq(x_org)
        uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
        emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)

        for sbmt_j in metadata:
            emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)
            with torch.no_grad():
                _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
            if len_pad == 0:
                uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
            else:
                uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            spect_vc.append(('{}x{}'.format(sbmt_i[0], sbmt_j[0]), uttr_trg))

    with open('results.pkl', 'wb') as handle:
        pickle.dump(spect_vc, handle)

    return spect_vc


def hifigan(args):
    generator = hifi_Generator(h).to(device)

    state_dict_g = load_checkpoint(args.hifigan_checkpoint, device)
    generator.load_state_dict(state_dict_g['generator'])

    # filelist = os.listdir(a.input_mels_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()

    data = pickle.load(open(args.input_mels_dir, 'rb'))
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

            output_file = os.path.join(args.output_dir, os.path.splitext(filename)[0] + '_generated_e2e.wav')
            # write(output_file, h.sampling_rate, audio)
            write(output_file, 16000, audio)
            # librosa.output.write_wav(output_file, audio, sr=16000)
            print(output_file)



def main(args):
    converted_pkl_exists = True
    if not converted_pkl_exists:
        # voice conversion
        spec_vc = inference(args)

    # vocoder
    hifigan(args)

    # with open('results.pkl', 'wb') as handle:
    #     pickle.dump(spect_vc, handle)

    # wavenet vocoder
    # spect_vc = pickle.load(open('results.pkl', 'rb'))
    # device = torch.device("cuda")
    # model = build_model().to(device)
    # checkpoint = torch.load("./checkpoint/wavenet/checkpoint_step001000000_ema.pth")
    # model.load_state_dict(checkpoint["state_dict"])
    #
    # for spect in spect_vc:
    #     name = spect[0]
    #     c = spect[1]
    #     print(name)
    #     waveform = wavegen(model, c=c)
    #     librosa.output.write_wav(name+'.wav', waveform, sr=16000)

    # hifigan vocoder




if __name__ == '__main__':

    print('Begin to infer......')

    parser = argparse.ArgumentParser()
    # autovc
    parser.add_argument('--autovc_checkpoint', default='./checkpoint/autovc/autovc.ckpt')
    parser.add_argument('--spec', default='metadata.pkl')
    # hifigan
    parser.add_argument('--input_mels_dir', default='/home/hongcz/alab/code/VF-VC/results_autovc.pkl')
    parser.add_argument('--output_dir', default='/home/hongcz/alab/code/VF-VC/hifigan/generated_files_from_mel')
    parser.add_argument('--hifigan_checkpoint', default='/home/hongcz/alab/code/VF-VC/checkpoint/hifigan/g_03280000')
    args = parser.parse_args()

    config_file = os.path.join(os.path.split(args.hifigan_checkpoint)[0], 'config.json')
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


    main(args)



