import os
import random

src = '/home/hongcz/alab/data/aishell3/train/wav16'
spklist = sorted(os.listdir(src))

trnlist = []
vallist = []

for spk in spklist:
    source = os.path.join(src, spk)
    wavlist = sorted(os.listdir(source))
    
    for wav in wavlist:
        trnlist.append(wav[:-4])


vallist = random.sample(trnlist, 64)

for i in vallist:
    trnlist.remove(i)


with open('/home/hongcz/alab/code/hifi_gan_master/aishell3_50/aishell3_50_trnlist.txt', 'w') as f1:
    for j in trnlist:
        f1.write(j)
        f1.write('\n')

with open('/home/hongcz/alab/code/hifi_gan_master/aishell3_50/aishell3_50_vallist.txt', 'w') as f2:
    for k in vallist:
        f2.write(k)
        f2.write('\n')


# root = '/home/hongcz/alab/data/VCTK-Corpus/wav16'
#
# spklist = sorted(os.listdir(root))
# with open('../feature/wavlist_VCTK.txt', 'w') as f:
#     for spk in spklist:
#         wavlist = sorted(os.listdir(os.path.join(root, spk)))
#         for wav in wavlist:
#             f.write(wav[:-4])
#             f.write('\n')
