import os
import random

src = '/home/hongcz/alab/feature/bnf_VCTK_50'

spklist = sorted(os.listdir(src))

trainlist = []
vallist = []

for spk in spklist:
    source = os.path.join(src, spk)
    wavlist = os.listdir(source)
    for wav in wavlist:
        trainlist.append(wav[:-4])


vallist = random.sample(trainlist, 64)
for j in vallist:
    trainlist.remove(j)

with open('wavlist_trn_VCTK_bnf_ge2e_50.txt', 'w') as f:
    for i in trainlist:
        f.write(i)
        f.write('\n')

with open('wavlist_val_VCTK_bnf_ge2e_50.txt', 'w') as f:
    for i in vallist:
        f.write(i)
        f.write('\n')

