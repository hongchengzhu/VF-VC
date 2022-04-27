from model_vc import Generator
import numpy as np
import soundfile as sf
import torch
import pickle
from vc_validation import validation
import matplotlib.pyplot as plt
import yaml


# data loading...
mel = torch.tensor(np.load('/home/hongcz/alab/feature/mel_hifigan_padding_alignment/LJ001-0003.npy')).to('cuda:0').unsqueeze(0)
content = pickle.load(open('/home/hongcz/alab/feature/wav2vec2_padding_alignment/LJ001-0003.pkl', "rb")).to('cuda:0')

# initialize...
# config file is in './config/config.yaml'
with open('./config/config_infer.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
model = Generator(config)
model.load_state_dict(torch.load('log/model/w2m-post/model155000.ckpt', map_location={'cuda:1': 'cuda:0'}))
model.to('cuda:0')
model.eval()

loss = {}
output = {}

validate = validation()

# pad because ConvTranspose
tag = mel.shape[1] % 4
# if tag != 0:

nonpadding = (mel.transpose(1, 2) != 0).float()[:, :]
nonpadding_mean = torch.mean(nonpadding, 1, keepdim=True)
nonpadding_mean[nonpadding_mean > 0] = 1

loss, output = model(mel, cond=content, loss=loss, output=output, nonpadding=nonpadding_mean, infer=True)
# self.val_output = output
input = output['recon_vae'].transpose(1, 2)
# inverse pad: recover
real_input = input * nonpadding

# show mel-spectrogram
spec = real_input.detach().cpu().numpy().squeeze(0)
plt.imshow(spec)
plt.show()
plt.savefig('inference_post_3_155000.wav.png')


vc_wav = validate.hifigan(real_input)

name = 'infer_post_3_155000.wav'
sf.write(name, vc_wav[0], samplerate=16000)


def plot_spectrogram(spectrogram):
    if not spectrogram.shape[0] == 80:
        spectrogram = spectrogram.T
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig

