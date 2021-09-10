import os,sys
sys.path.append('/Library/Python/2.7/site-packages/')
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages')
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))

import numpy as np
from pydub import AudioSegment
from librosa import resample,load, to_mono
import utilFunctions as UF

if __name__ == '__main__':
    
    # 1: resample wav
    # 2: convert mp3 to wav
    exp = 2

    if exp == 1:
        fd_path = '../../sounds/midi_synth/'
        files = os.listdir(fd_path)
        for f in files:
            if f[-3:] == 'wav':
                x, fs = load(fd_path+f)
                y = to_mono(resample(x,fs,44100))
                UF.wavwrite(y, 44100, fd_path+f)
    
    elif exp == 2:
        mp3_path = '../../sounds/phiharmonia_db/'
        wav_path = '../../sounds/phiharmonia/'

        # instruments = ['flute','oboe','clarinet','saxophone','french horn','trumpet','violin','cello']
        instruments = ['cello']

        for ins in instruments:
            for f in os.listdir(mp3_path+ins+'/'):
                try:
                    div = [pos for pos, char in enumerate(f) if char == '_']
                    length = f[div[1]+1:div[2]]
                    articulation = f[div[3]+1:f.find('.')]
                    if length in ['1','05','15'] and articulation in ['normal','arco-normal']:
                        sound = AudioSegment.from_mp3(mp3_path+ins+'/'+f)
                        sound.export(wav_path+ins+'/'+f[:-3]+'wav',format = "wav")
                except:
                    print(f,'Failed!')
                
            
            print(ins,'Done!')

