import os
import shutil

folders = ['filter', 'ge2e']

spklist = sorted(os.listdir('/home/hongcz/alab/data/VCTK-Corpus/wav16'))

source = '/home/hongcz/alab/data/VCTK-Corpus/wav16'
target = './vctk-wavs'


with open('vclist_or.txt', 'w') as f:
    wavlist = sorted(os.listdir('./vctk-wavs_lps'))
    for i in wavlist:
        f.write(i[:-4] + ' bonafide')
        f.write('\n')



for spk in spklist:
    wavlist = sorted(os.listdir(os.path.join(source, spk)))[:10]
    for wav in wavlist:
        shutil.copyfile(os.path.join(source, spk, wav), os.path.join(target, wav))


print(1)


with open('vclist_or.txt', 'w') as f1:
    for spk in spklist:
        print(1)



with open('vclist.txt', 'w') as f:
    for folder in folders:
        filelist = sorted(os.listdir(folder))
        for i in filelist:
            to_write = i+ ' spoof'
            f.write(to_write)
            f.write('\n')
        
