import os
import time
from model_vc import Generator
import numpy as np
import soundfile as sf
import torch
import pickle
from vc_validation import validation
import matplotlib.pyplot as plt
import yaml
from torch.nn.utils.rnn import pad_sequence



# data loading...
# mel = torch.tensor(np.load('/home/hongcz/alab/feature/mel_VCTK/p225/p225_001.npy')).to('cuda:0').unsqueeze(0)
# content = pickle.load(open('/home/hongcz/alab/feature/wav2vec2_VCTK/p225/p225_001.pkl', "rb")).to('cuda:0')
# spk_emb = torch.tensor(np.load('/home/hongcz/alab/feature/speaker_embedding_VCTK/p228/p228.npy')).to('cuda:0').unsqueeze(0)

# initialize...
# config file is in './config/config.yaml'
with open('./config/config_infer.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
model = Generator(config)
model.load_state_dict(torch.load('log/model/w2m-vctk-5-1/model430000.ckpt', map_location={'cuda:1': 'cuda:0'}))
model.to('cuda:0')
model.eval()

loss = {}
output = {}

validate = validation()
c_src = '/home/hongcz/alab/feature/wav2vec2_VCTK/'
emb_src = '/home/hongcz/alab/feature/spk_emb_VCTK_ge2e'
uttlist = ['p225_001', 'p226_002', 'p227_003', 'p228_004', 'p229_005',
           'p230_006', 'p231_007', 'p232_008', 'p233_009', 'p234_010']

# inference test
uttlist = []
speakerlist = sorted(os.listdir(emb_src))
for speaker in speakerlist[:100]:
    uttlist.append(sorted(os.listdir(os.path.join(c_src, speaker)))[0][:-4])


spklist = sorted(os.listdir(emb_src))[:10]
# tgt_emb = torch.zeros([10, 256])

# for i in range(len(uttlist)):
for k in spklist:
    torch.cuda.empty_cache()
    # content = torch.FloatTensor(np.load(os.path.join(c_src, uttlist[i][:-4], uttlist[i]+'.npy'))).to('cuda:0')
    content = []
    for i in range(len(uttlist)):
        content.append(pickle.load(open(os.path.join(c_src, uttlist[i][:-4], uttlist[i]+'.pkl'), "rb")).squeeze(0).to('cuda:0'))
        # content = content.repeat(10, 1, 1)
    content = pad_sequence(content, padding_value=0).permute(1, 0, 2)

    torch.cuda.empty_cache()

    # for j in range(109):
    #     tgt_emb[j, :] = torch.FloatTensor(np.loadtxt(os.path.join(emb_src, spklist[j], spklist[j]+'.txt'), delimiter=',')).to('cuda:0')
    tgt_emb = torch.FloatTensor(np.loadtxt(os.path.join(emb_src, k, k + '.txt'), delimiter=',')).to('cuda:0')
    tgt_emb = tgt_emb.repeat(100, 1)
    # tgt_emb = tgt_emb.to('cuda:0')
    # tgt_emb1 = torch.FloatTensor(np.loadtxt(os.path.join(emb_src, uttlist[i][:-4], uttlist[i][:-4]+'.txt'), delimiter=',')).to('cuda:0')
    # tgt_emb2 = torch.FloatTensor(np.loadtxt(os.path.join(emb_src, uttlist[(i+1)%4][:-4], uttlist[(i+1)%4][:-4] + '.txt'), delimiter=',')).to('cuda:0')
    # tgt_emb = torch.stack((tgt_emb1, tgt_emb2), dim=0)

# content = pickle.load(open(c_src, "rb")).to('cuda:0')
# content = np.load(c_src)

# pad because ConvTranspose
    tag = content.shape[1] % 4
# if tag != 0:

    nonpadding = (content.transpose(1, 2) != 0).float()[:, :]
    nonpadding_mean = torch.mean(nonpadding, 1, keepdim=True)
    nonpadding_mean[nonpadding_mean > 0] = 1

    loss, output = model(None, cond=content, loss=loss, spk_emb=tgt_emb, output=output, nonpadding=nonpadding_mean, infer=True)
    # self.val_output = output
    input_vae = output['recon_vae'].transpose(1, 2)
    input_postflow = output['recon_post_flow'].transpose(1, 2)
    # inverse pad: recover
    real_input_vae = input_vae * nonpadding_mean
    real_input_postflow = input_postflow * nonpadding_mean

    # real_input = torch.cat((real_input_vae, real_input_postflow), dim=0)

    real_input = real_input_postflow

    vc_wav = []
    for input_index in range(len(real_input)):
        vc_wav = validate.hifigan(real_input[input_index, :, :].unsqueeze(0))
        torch.cuda.empty_cache()

    # name = [uttlist[i][:-4] + '_' + uttlist[i][:-4], uttlist[i][:-4] + '_' + uttlist[(i+1)%4][:-4]]
        # name.append(os.path.join('spoof/ge2e', uttlist[i] + '_to_' + spklist[k]))
        name = '/home/hongcz/alab/code/VF-VC-5-1/id_test/id_infer/ge2e/' + uttlist[input_index][:-4] + '_to_' + k

        # sf.write(name[j]+'_vae.wav', vc_wav[j], samplerate=16000)
        sf.write(name + '.wav', vc_wav[0], samplerate=16000)
        torch.cuda.empty_cache()
    # time.sleep(3)


