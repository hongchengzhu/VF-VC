from model_vc import Generator
import numpy as np
import soundfile as sf
import torch
import pickle
from vc_validation import validation
import matplotlib.pyplot as plt


# data loading...
mel = torch.tensor(np.load('./feature/mel_s3prl/LJ001-0001.npy')).to('cuda:0').unsqueeze(0)
content = pickle.load(open('./feature/wav2vec2/LJ001-0001.pkl', "rb"))

# initialize...
model = Generator()
model.load_state_dict(torch.load('./log/model/w2m-non-parallel/VF-VC_VAE_59989.ckpt'))
model.to('cuda:0')
model.eval()

loss = {}
output = {}

validate = validation()

# pad because ConvTranspose
tag = mel.shape[1] % 4
# if tag != 0:
mel_padding = torch.zeros([1, mel.shape[1] + (4 - tag), 80]).to('cuda:0')
content_padding = torch.zeros([1, content.shape[1] + (4 - tag), 768]).to('cuda:0')

mel_padding[:, :mel.shape[1], :] = mel
content_padding[:, :content.shape[1], :] = content

nonpadding = (mel_padding.transpose(1, 2) != 0).float()[:, :]
nonpadding_mean = torch.mean(nonpadding, 1, keepdim=True)
nonpadding_mean[nonpadding_mean > 0] = 1

loss, output = model(mel_padding, cond=content_padding,
                                        loss=loss, output=output,
                                        nonpadding=nonpadding_mean, infer=True)
# self.val_output = output
input = output['x_recon'].transpose(1, 2)
# inverse pad: recover
real_input = torch.zeros([1, 80, input.shape[-1] - tag]).to('cuda:0')
real_input = input[:, :, :input.shape[-1] - tag]

# show mel-spectrogram
spec = real_input.detach().cpu().numpy().squeeze(0)
plt.imshow(spec)
plt.show()
plt.savefig('inference_59989.wav.png')


vc_wav = validate.hifigan(real_input)

name = 'inference.wav'
sf.write(name, vc_wav, samplerate=16000)

