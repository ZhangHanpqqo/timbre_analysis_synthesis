##########################################################################
#    class: sound_info                                                   #
#    attributes: x, path, hfreq, hmag, hphase, stocEnv, sdInfo           #
#    functions:                                                          #
##########################################################################

import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../transformations/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../modification/'))
sys.path.append('/Library/Python/2.7/site-packages/')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from scipy.signal import get_window
from joblib import load
from copy import deepcopy

import utilModi as UM
import utilFunctions as UF
import utilModiGUI as UMG
import timbreModi as TM
import harmonicModel as HM
import stft as STFT

class sound_info:
    def __init__(self, path, pitch, N=8192, M=8192, H=256, nH=40, window='blackmanharris'):
        # get the sound from file path
        if isinstance(pitch, str):
            minf0 = UM.pitch2freq(pitch)-50
        else:
            minf0 = pitch-50
        maxf0 = minf0 + 100
        self.path = path

        
        # get the dictionary of sound info
        self.get_sdInfo(pitch, nH, minf0, maxf0, N, M, H, window)
        self.sdInfo_org = deepcopy(self.sdInfo)  # always preserve the original sound info

        # get the synthesized sound, frequency, magnitude, and phase
        self.get_synInfo()

    def update(self, path, pitch, N, M, H, nH, window):
        # get the sound from file path
        if isinstance(pitch, str):
            minf0 = UM.pitch2freq(pitch)-50
        else:
            minf0 = pitch-50
            pitch = None
        maxf0 = minf0 + 100
        self.path = path

        
        # get the dictionary of sound info
        self.get_sdInfo(pitch, nH, minf0, maxf0, N, M, H, window)

        # get the synthesized sound, frequency, magnitude, and phase
        self.get_synInfo()

    def get_sdInfo(self, pitch, nH, minf0, maxf0, N, M, H, window):
        Ns=512
        self.sdInfo = {
                'instrument': '',
                'pitch': pitch,
                'source': '',
                'index': '',
                'nH': nH,
                'nF':0,
                'windowSize':M,
                'FFTLenAna':N,
                'FFTLenSyn':Ns,
                'hopSize':H,
                'window': window
                }

        self.x, fs, self.hfreq, self.hmag, self.hphase, self.stocEnv, self.sdInfo = UM.sound_info_clct(self.path, self.sdInfo, nH=nH, minf0=minf0, maxf0=maxf0, M=M, N=N, Ns=Ns, H=H, window=window)

    def get_synInfo(self, synFreq=True, synMag=True, synPhase=True):
        self.y, self.yh, _,  self.hfreqSyn, self.hmagSyn, self.hphaseSyn = TM.sound_syn_from_para(self.sdInfo)
    
    def get_sdFt(self):
        self.sdFt = UM.dict2vector(self.sdInfo)

    def get_sdInfo_from_sdFt(self, nF=1000, silenceSt=10, silenceEd=10, meanMax=-100, nH=40):
        self.sdInfo = UM.vector2dict(self.sdFt[0],self.sdFt[1:],nF,silenceSt,silenceEd,meanMax,nH)
        self.get_synInfo()

    def get_sdInfo_morph(self, sdInfo_ref, morph_rate, duration, intensity):
        self.sdInfo, self.sdFt = UM.sound_morphing(self.sdInfo, sdInfo_ref, morph_rate, duration, intensity)

    def get_class(self):
        clf_path = '../modification/model/rf_8ins_'+ str(self.sdInfo['nH']) +'harm.joblib'
        clf = load(clf_path)
        ins = ['flute','oboe','clarinet','saxophone','french horn','trumpet','violin','cello']
        self.class_res = ins[clf.predict(np.array([self.sdFt[1:]]))[0]]

    def display_original_sound(self):
        UF.wavplay(self.path)

    def display_synth_sound(self, outputFile='output_sounds/syn.wav'):
        UF.wavwrite(self.y, self.sdInfo['fs'],outputFile)
        UF.wavplay(outputFile)

    def display_original_plot_3d(self):
        t = np.arange(self.sdInfo['nF']) * self.sdInfo['hopSize'] / self.sdInfo['fs']
        UM.plot_spec3d(self.hfreq, self.hmag, t, 1)

    def display_synth_plot_3d(self):
        t = np.arange(self.sdInfo['nF']) * self.sdInfo['hopSize'] / self.sdInfo['fs']
        UM.plot_spec3d(self.hfreqSyn, self.hmagSyn, t, 1)

    def display_original_plot_harm(self, harmNo):
        t = np.arange(self.sdInfo['nF']) * self.sdInfo['hopSize'] / self.sdInfo['fs']
        
        plt.subplot(1,2,1)
        plt.plot(t, self.hmag[:,harmNo])
        plt.xlabel('time(s)')
        plt.ylabel('magnitude(dB)')
        plt.title('Harmonic #'+str(harmNo)+" time-magnitude")

        plt.subplot(1,2,2)
        plt.plot(t, self.hfreq[:,harmNo])
        plt.xlabel('time(s)')
        plt.ylabel('frequency(Hz)')
        plt.title('Harmonic #'+str(harmNo)+" time-frequency")

        plt.show()

    def display_synth_plot_harm_dragPoint(self, harmNo, harmName):
        t = np.arange(self.sdInfo['nF']) * self.sdInfo['hopSize'] / self.sdInfo['fs']
        
        if harmName == 'magnitude':
            UMG.point_drag_axes(t, self.hmag[:,harmNo], harmName, harmNo, 
                    self.sdInfo['magADSRIndex'][:,harmNo], 
                    self.sdInfo['magADSRValue'][:,harmNo], self.sdInfo['magADSRN'][:,harmNo])
        
        elif harmName == 'frequency':
            UMG.point_drag_axes(t, self.hfreq[:,harmNo], harmName, harmNo,
                    self.sdInfo['magADSRIndex'][:,harmNo], 
                    np.ones(4)*(self.sdInfo['f0']+self.sdInfo['f0']*harmNo*self.sdInfo['freqInterval']+self.sdInfo['freqMean'][harmNo]),
                    init_variance=self.sdInfo['freqVar'][harmNo]+self.sdInfo['freqVarRate'])        

    def get_spectrogram(self, source='original'):
        if source == 'original':
            mX, pX = STFT.stftAnal(self.x, get_window(self.sdInfo['window'], self.sdInfo['windowSize']), self.sdInfo['FFTLenAna'], self.sdInfo['hopSize'])
        else:
            mX, pX = STFT.stftAnal(self.y, get_window(self.sdInfo['window'], self.sdInfo['windowSize']), self.sdInfo['FFTLenAna'], self.sdInfo['hopSize'])
        return mX

if __name__ == '__main__':
    file_path = '../../sounds/A4/trumpet-A4.wav'
    sd = sound_info(file_path, 'A4')
    sd.display_synth_plot_harm_dragPoint(0,'magnitude')
    


