import soundfile as sf
import os
from tqdm import tqdm
import librosa
import threading

root = '/home/hongcz/alab/data/VCTK-Corpus/wav48'
target = '/home/hongcz/alab/data/VCTK-Corpus/wav16'
sr_ = 48000
_sr = 16000

global spklist
spklist = sorted(os.listdir(root))


def resample(index):
    spk = spklist[index]
    source = os.path.join(root, spk)
    tgt = os.path.join(target, spk)
    if not os.path.exists(tgt):
        os.mkdir(tgt)
    wavlist = os.listdir(source)
    for wav in tqdm(wavlist):
        waveform, _ = sf.read(os.path.join(source, wav))
        waveform_ = librosa.resample(waveform, sr_, _sr)
        sf.write(os.path.join(tgt, wav), waveform_, samplerate=16000)
    print('{} process over!'.format(spk))


for i in range(33, 109, 4):
    t1 = threading.Thread(target=resample, args=(i, ))
    t2 = threading.Thread(target=resample, args=(i+1, ))
    t3 = threading.Thread(target=resample, args=(i+2, ))
    t4 = threading.Thread(target=resample, args=(i+3, ))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()

