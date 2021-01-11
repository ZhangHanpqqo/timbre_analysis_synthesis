import os,sys
sys.path.append('/Library/Python/2.7/site-packages/')
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))

import numpy as np
from librosa import resample,load, to_mono
import utilFunctions as UF

if __name__ == '__main__':
    fd_path = '../../sounds/flute_acoustic/flute_acoustic_test/'
    files = os.listdir(fd_path)
    for f in files:
        if f[-3:] == 'wav':
            x, fs = load(fd_path+f)
            y = to_mono(resample(x,fs,44100))
            UF.wavwrite(y, 44100, fd_path+f)




