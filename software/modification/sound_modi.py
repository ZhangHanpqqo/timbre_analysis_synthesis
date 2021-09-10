import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../transformations/'))
sys.path.append('/Library/Python/2.7/site-packages/')

import numpy as np
import matplotlib.pyplot as plt
import copy
import math

import utilFunctions as UF
import utilModi as UM
import instrument_feature as IF
import timbreModi as TM

def get_sdInfo(file_path, pitch, instrument = None, source = None, index = None, nF=0, nH=40, M=4096, N=8192, Ns=512, H=256):
    # get collecting parameters
    if isinstance(pitch, str):
        pitch = UM.pitchname2num(pitch)
    if pitch < 72:
        M = 4096
    else:
        M = 2048
    N = M*2
    frequency = UM.pitch2freq(pitch)
    minf0 = frequency - 50
    maxf0 = frequency + 50

    sdInfo = {
        'instrument':instrument,
        'pitch':pitch,
        'source':source,
        'index':index,
        'nH':nH,
        'nF':nF,
        'FFTLenAna':N,
        'FFTLenSyn':Ns,
        'hopSize':H
    }

    
    sdInfo = TM.sound_info_clct(file_path, sdInfo, nH, minf0, maxf0, M, N, Ns, H)
    
    return sdInfo
    
if __name__ == '__main__':
    
    # 1: sound morphing
    # 2: tutti
    exp = 1

    if exp == 1:
        # file_path1 = '../../sounds/phiharmonia/violin/violin_A5_15_forte_arco-normal.wav'
        # file_path2 = '../../sounds/phiharmonia/flute/flute_A5_15_forte_normal.wav'
        # sd_path1 = 'result/features/phiharmonia/violin/violin-81-4.json'
        # sd_path2 = 'result/features/phiharmonia/flute/flute-81-4.json'

        file_path1 = '../../sounds/phiharmonia/clarinet/clarinet_A4_15_forte_normal.wav'
        file_path2 = '../../sounds/phiharmonia/violin/violin_A4_15_fortissimo_arco-normal.wav'
        # sd_path1 = 'result/features/phiharmonia/trumpet/trumpet-69-4.json'
        # sd_path2 = 'result/features/phiharmonia/clarinet/clarinet-69-4.json'

        ins1 = 'clarinet'
        ins2 = 'violin'
        pitch1 = 'A4'
        pitch2 = 'A4'

        sdInfo1 = get_sdInfo(file_path1, pitch1, ins1, index = '1'+ins1)
        sdInfo2 = get_sdInfo(file_path2, pitch2, ins2, index = '1'+ins2)
        # sdInfo2 = UM.read_features(sd_path2,fm=1)
        # sdInfo = UM.read_features(sd_path1,fm=1)

        morph_rate = [1, 0.9, 0.75, 0.5, 0.25, 0.1, 0]
        
        for mr in morph_rate:
            sdInfo = copy.deepcopy(sdInfo1)
            sdInfo['instrument'] = 'synth'
            sdInfo['index'] = str(mr)+ins1+'+'+str(1-mr)+ins2
            for key in sdInfo1:
                if key not in ['instrument', 'pitch', 'source', 'index', 'nH', 'FFTLenAna', 'FFTLenSyn', 'hopSize', 'fs', 'stocEnv']:
                    sdInfo[key] = mr*sdInfo1[key] + (1-mr)*sdInfo2[key]
                    if key in ['nF','freqSmoothLen']:
                        sdInfo[key] = int(sdInfo[key])
                    elif key in ['magADSRIndex']:
                        sdInfo[key] = np.array(sdInfo[key],dtype=int)
            sdInfo['nF'] = int(sdInfo['nF'])

            y, yh, yst, hfreqSyn, hmagSyn, hphaseSyn = TM.sound_syn_from_para(sdInfo)
            outputFile = 'output_sounds/morph_'+sdInfo['index']+'.wav'
            UF.wavwrite(y, sdInfo['fs'], outputFile)
            # UF.wavplay(outputFile)


    elif exp == 2:
        # file_path1 = '../../sounds/phiharmonia/violin/violin_A4_15_fortissimo_arco-normal.wav'
        # file_path1 = '../../sounds/phiharmonia/flute/flute_A4_15_forte_normal.wav'
        # file_path2 = '../../sounds/phiharmonia/clarinet/clarinet_A4_15_forte_normal.wav'
        file_path1 = 'output_sounds/combine/combine_flute+clarinet.wav'
        file_path2 = 'output_sounds/combine/combine_violin+trumpet.wav'

        ins1 = 'flute+clarinet'
        ins2 = 'violin+trumpet'

        (fs,x1) = UF.wavread(file_path1)
        (fs,x2) = UF.wavread(file_path2)

        if x1.size <= x2.size:
            xl = x2
            xs = x1
        else:
            xl = x1
            xs = x2

        # for i in range(len(xs)):
        #     xs[i] = (xs[i] + xl[math.floor(i/len(xs)*len(xl))])/2

        x = (xl+np.concatenate((xs,np.zeros(xl.size-xs.size))))/2

        outputFile = 'output_sounds/combine/combine_'+ins1+'+'+ins2+'.wav'
        UF.wavwrite(x, fs, outputFile)
        UF.wavplay(outputFile)
    
