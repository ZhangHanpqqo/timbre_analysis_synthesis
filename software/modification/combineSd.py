import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../transformations/'))
sys.path.append('/Library/Python/2.7/site-packages/')

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.fft import fft
from pyAudioAnalysis import audioAnalysis as aa
import json
import copy

import utilModi as UM
import timbreModi as TM
import utilFunctions as UF
import spsModel as SPS

def combine_sound(folder_path, length=None):
    if not os.path.exists(folder_path):     # reporting error if the path does not exist
        print('folder is not found!')
        return
     
    files = os.listdir(folder_path)
    sound_num = len(files)
    first_flag = 0
    if sound_num == 0:
        return
    else:
        for f in files:
            if first_flag == 0:
                (fs, x) = UF.wavread(folder_path+f)
                x = x/sound_num
                first_flag = 1
            elif f[-3:] == 'wav': 
                (_,x_cur) = UF.wavread(folder_path+f)

                # match the length of the combined sound and the new sound to be merged
                x_len = x.shape[0]
                x_cur_len = x_cur.shape[0]
                if x_len > x_cur_len: 
                    x_cur = np.concatenate((x_cur, np.zeros(x_len-x_cur_len)))
                else:
                    x = np.concatenate((x, np.zeros(x_cur_len-x_len)))

                x = x + x_cur/sound_num
    
    if length is not None and len(x) > length:
        x = x[:length]
                
    return x, fs

def sps_ana(file_path):
    # get sound
    (fs, x) = UF.wavread(file_path)
        
    # parameters for sps
    window = 'blackmanharris'
    M = 4096
    w = sig.get_window(window,M)
    N = 4096
    H = 256
    t = -80
    minSineDur = 0.08
    maxnSine = 20
    freqDevOffset = 20
    freqDevSlope = 0.001
    stocf = 0.2

    # apply sps model
    tfreq, tmag, tphase, stocEnv = SPS.spsModelAnal(x, fs, w, N, H, t, minSineDur, maxnSine, freqDevOffset, freqDevSlope, stocf)

    return tfreq, tmag, tphase, stocEnv
        

if __name__ == '__main__':
    
    exp = 1

    # 0: combine sounds and show spectrogram
    # 1: test HpS and SpS model on combined sounds

    if exp == 0:
        # combine the sound
        folder_path = '../../sounds/OrchideaSOL/sds4/'
        x, fs = combine_sound(folder_path)
        UF.wavwrite(x,fs,folder_path+'combined-full.wav')
        # UF.wavplay(folder_path+'combined.wav')

        # check spectrogram
        UM.plot_spectrogram(x)


    if exp == 1:
        # get target combined sound
        file_path = '../../sounds/OrchideaSOL/sds0/combined-full.wav'

        tfreq, tmag, tphase, stocEnv = sps_ana(file_path)       
        
        print('freq',np.average(tfreq,axis=0), 'mag',tmag)
        plt.imshow(np.transpose(tfreq))
        plt.show()
        # UM.plot_spec3d(tfreq, tmag, np.arange(tfreq.shape[0]))
        

