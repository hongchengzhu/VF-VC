# import upstreaming task model
import os
import torch
import pickle
import s3prl.hub as hub
import soundfile as sf
from tqdm import tqdm


device = 'cuda:0'

wav2vec_model = getattr(hub, 'wav2vec2')()
wav2vec_model = wav2vec_model.to(device)

# wavs = [torch.randn(180000, dtype=torch.float).to(device)]


with torch.no_grad():
    wav_dir = '/home/hongcz/alab/data/LJSpeech-1.1/test/LJ014'
    wavs = []
    wav_list = os.listdir(wav_dir)
    for wavname in tqdm(sorted(wav_list)):
        wav, _ = sf.read(os.path.join(wav_dir, wavname))
        wav = torch.tensor(wav, dtype=torch.float32).to(device).unsqueeze(0)
        # wavs.append(wav)
        ret = wav2vec_model(wav)['last_hidden_state']
        to_name = '/home/hongcz/alab/feature/wav2vec2/' + wavname[:-4] + '.pkl'
        with open(to_name, 'wb') as f:
            pickle.dump(ret, f)
    print(1)
