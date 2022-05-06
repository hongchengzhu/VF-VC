import os
import shutil


source = '/mnt/datasets/VCTK-Corpus/wav16'
target = '/mnt/datasets/VCTK-Corpus/wavs'
if not os.path.exists(target):
    os.mkdir(target)

spklist = os.listdir(source)
for spk in spklist:
    src = os.path.join(source, spk)
    wavlist = os.listdir(src)
    for wav in wavlist:
        shutil.copyfile(os.path.join(src, wav), os.path.join(target, wav))
