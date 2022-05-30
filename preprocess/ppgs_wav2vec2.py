import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import AutoProcessor, AutoModelForPreTraining
from datasets import load_dataset
import soundfile as sf
import torch
import soundfile as sf
import numpy as np
import pickle
import threading
from tqdm import tqdm
import time

# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to('cuda:0')

# load dummy dataset and read soundfiles
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

source = '/home/hongcz/alab/data/VCTK-Corpus/wav16'
target = '/home/hongcz/alab/feature/ppg_VCTK'
content = '/home/hongcz/alab/feature/mel_VCTK'
if not os.path.exists(target):
    os.mkdir(target)

spklist = sorted(os.listdir(source))


def ppg_extract(index):
    # tokenize
    spk = spklist[index]
    src = os.path.join(source, spk)

    wavlist = os.listdir(src)
    for wav in tqdm(wavlist):
        input = torch.tensor(sf.read(os.path.join(src, wav))[0], dtype=torch.float32).to('cuda:0').unsqueeze(0)
        # retrieve logits
        logits = model(input).logits

        c_len = np.load(os.path.join(os.path.join(content, spk), wav[:-4]+'.npy')).shape[0]

        tag = logits.shape[1] % 4
        if tag != 0:
            tmp = torch.zeros([1, logits.shape[1] + (4 - tag), logits.shape[2]])
            tmp[:, :logits.shape[1], :] = logits
            logits = tmp

        l_len = logits.shape[1]
        if c_len > l_len:
            tmp = torch.zeros([1, c_len, logits.shape[2]])
            tmp[:, :l_len, :] = logits
            logits = tmp
        if c_len < l_len:
            logits = logits[:, :c_len, :]

        # take argmax and decode
        # predicted_ids = torch.argmax(logits, dim=-1)
        # transcription = processor.batch_decode(predicted_ids)
        # print(transcription)

        tgt = os.path.join(target, spk)
        if not os.path.exists(tgt):
            os.mkdir(tgt)
        to_name = os.path.join(tgt, wav[:-4])
        np.save(to_name, logits.cpu().detach().numpy().astype('float32'), allow_pickle=False)


for i in range(0, 109, 1):
    ppg_extract(i)
    torch.cuda.empty_cache()


# for i in range(8, 109, 8):
#     t1 = threading.Thread(target=ppg_extract, args=(i, ))
#     t2 = threading.Thread(target=ppg_extract, args=(i+1, ))
#     t3 = threading.Thread(target=ppg_extract, args=(i+2, ))
#     t4 = threading.Thread(target=ppg_extract, args=(i+3, ))
#     t5 = threading.Thread(target=ppg_extract, args=(i+4, ))
#     t6 = threading.Thread(target=ppg_extract, args=(i+5, ))
#     t7 = threading.Thread(target=ppg_extract, args=(i+6, ))
#     t8 = threading.Thread(target=ppg_extract, args=(i+7, ))
#
#     t1.start()
#     t2.start()
#     t3.start()
#     t4.start()
#     t5.start()
#     t6.start()
#     t7.start()
#     t8.start()
#
#     t1.join()
#     t2.join()
#     t3.join()
#     t4.join()
#     t5.join()
#     t6.join()
#     t7.join()
#     t8.join()
#
#     torch.cuda.empty_cache()
