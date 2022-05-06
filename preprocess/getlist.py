import os
import random

src = '../feature/wavlist_VCTK.txt'
with open(src, 'r') as f:
    wavlist = f.readlines()

    vallist = sorted(random.sample(wavlist, 64))
    trnlist = []
    for i in wavlist:
        if i not in vallist:
            trnlist.append(i)
    with open('../feature/wavlist_trn_VCTK.txt', 'w') as f1:
        for j in trnlist:
            f1.write(j)
    with open('../feature/wavlist_val_VCTK.txt', 'w') as f2:
        for k in vallist:
            f2.write(k)


# root = '/home/hongcz/alab/data/VCTK-Corpus/wav16'
#
# spklist = sorted(os.listdir(root))
# with open('../feature/wavlist_VCTK.txt', 'w') as f:
#     for spk in spklist:
#         wavlist = sorted(os.listdir(os.path.join(root, spk)))
#         for wav in wavlist:
#             f.write(wav[:-4])
#             f.write('\n')
