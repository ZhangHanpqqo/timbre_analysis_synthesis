#!/usr/bin/env python
# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../transformations/'))
sys.path.append('/Library/Python/2.7/site-packages/')

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy.signal as sig
import scipy.optimize as opt

import wave
# import pyaudio

import stft


## get audio data
def get_sound_data(file_path):
    sd = wave.open(file_path,'rb')
    paras = sd.getparams()
    nchannels, sampwidth, framerate, nframes = paras[:4]
    sd_data_str = sd.readframes(nframes)
    sd_data = np.fromstring(sd_data_str, dtype=np.short)

    sd.close()
    return sd_data, nchannels, sampwidth, framerate, nframes


## play the sound
# def play_sound(sd_data, nchannels, sampwidth, framerate):
#
#     p = pyaudio.PyAudio()
#
#     stream = p.open(format=p.get_format_from_width(sampwidth),
#                        channels=nchannels,
#                        rate=framerate,
#                        output=True)
#
#     stream.write(sd_data)
#
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
    

## plot the shape
def plot_wave(sd_data, rate, length):
    timeAxid = np.arange(1,length+1) * 1./ rate
    #sd_data_abs = np.abs(sd_data)
    fig = plt.figure()
    plt.plot(timeAxid, sd_data)
    fig.show()
    return 0

## find spectrum
#def find_spectrum(sd_data, framerate, display = 0):
#    N_fft = 1024 # length of fft window
#    #fig1 = plt.figure()
#    spectrum,freqs,ts, fig = plt.specgram(sd_data,NFFT=N_fft,Fs=framerate,cmap=cm.coolwarm)
#    freqs = freqs[:,np.newaxis]
#    ts = ts[:,np.newaxis]
#    #fig1.show()
#
#    # 3D display
#    fig2 = plt.figure()
#    if display == 1:
#        y = np.repeat(freqs,ts.shape[0],axis=1)
#        x = np.repeat(ts,freqs.shape[0],axis=1).T
#        z = spectrum
#        ax = plt.axes(projection="3d")
#        ax.plot_surface(x,y,z,cmap=cm.coolwarm)
#        ax.axis()
#        fig2.show()
#        
#    return spectrum,freqs,ts

## find spectrum
def find_spectrum(sd_data, framerate, display = 0):
    window_size = 1024
    window = sig.get_window('hamming',window_size)
    FFT_size = 1024
    Hop_size = 512
    
    mX, pX = stft.stftAnal(sd_data, window,FFT_size,Hop_size) # get amplitude spectrum and phase spectrum
    spectrum = mX.transpose()
    phase = pX.transpose()
    
    freq_frm = spectrum.shape[0]
    freqs = np.arange(freq_frm)*(float(framerate)/FFT_size)
    freqs = freqs[:,np.newaxis]
    
    time_frm = spectrum.shape[1]
    ts = (np.arange(time_frm)+0.5)*(Hop_size/float(framerate))
    ts = ts[:,np.newaxis]
    
    if display == 1:
        fig1 = plt.figure()
        plt.subplot(2,1,1)
        plt.pcolormesh(ts[:,0], freqs[:,0], spectrum, cmap=cm.coolwarm)
        plt.xlabel('time (sec)')
        plt.ylabel('frequency (Hz)')
        plt.title('magnitude spectrogram')
        
        plt.subplot(2,1,2)
        plt.pcolormesh(ts[:,0], freqs[:,0], phase, cmap=cm.coolwarm)
        plt.xlabel('time (sec)')
        plt.ylabel('frequency (Hz)')
        plt.title('phase spectrogram (derivative)')
         
        fig1.show()

    # 3D display
    fig2 = plt.figure()
    if display == 2:
        y = np.repeat(freqs,ts.shape[0],axis=1)
        x = np.repeat(ts,freqs.shape[0],axis=1).T
        z = spectrum
        ax = plt.axes(projection="3d")
        ax.plot_surface(x,y,z,cmap=cm.coolwarm)
        ax.axis()
        fig2.show()
        
    return spectrum,phase,freqs,ts


## frequency domain
# find fundamental frequency and partials
def find_partial(spectrum, freqs, display = 0):
    freq_sum = np.sum(spectrum,axis=1)
    freq_peak_ind = sig.find_peaks(freq_sum,height=4e3)
    freq_peak_ind = freq_peak_ind[0]
    freq_peak = freqs[freq_peak_ind]
    
    # find amplitude of each partial
    partial = []
    for i in freq_peak_ind:
        amp = spectrum[i,:]
        partial.append(amp)

    partial = np.array(partial)
    #N_pt = partial.shape[0]

    ## plot partials
    if display == 1:
        ax_partial = plt.axes(projection="3d")
        fig = plt.figure()
        for i in range(partial.shape[0]):
            ax_partial.plot(ts[:,0],freq_peak[i]*np.ones((ts.shape[0],1))[:,0],partial[i,:].T,)
            ax_partial.set_ylim([0,4000])
        fig.show()
    return freq_peak, partial

## time domian
def smooth(y, box_pts=20):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# find adsr envelop
def find_adsr(x):
    # find four time-value pairs: soa, eoa, sor, eor
    x_max = np.max(x)
    # soa: 10% of max from start
    # eoa: 90% of max from start
    fg = 0
    for i in range(x.shape[0]):
        if x[i] >= 0.9*x_max:
            eoa_ind = i
            break
        elif x[i] >= 0.1*x_max and fg is 0:
            soa_ind = i
            fg = 1

    # sor: 10% of max from end
    # eor: 90% of max from end
    fg = 0
    for i in range(x.shape[0]-1,0,-1):
        if x[i] >= 0.7*x_max:
            eor_ind = i
            break
        elif x[i] >= 0.1*x_max and fg is 0:
            sor_ind = i
            fg = 1

    return [soa_ind,eoa_ind,sor_ind,eor_ind]


## find curve model for partials
def curve(n,values,x):
    if x.shape[0] == 0:
        return 0
    else:
        v0 = values[0][0]
        v1 = values[-1][0]
        return v0+(v1-v0)*np.power((1-np.power((1-(x-x.min())/(x.max()-x.min())),n)),1./n)

def curve_err(n,values,x):
    if x.shape[0] == 0:
        return 0
    else:
        return (values-curve(n,values,x))[:,0]

def find_opt_n(values, x, n0 = 1):
    if x.shape[0] == 0:
        return 1
    else:
        res = opt.least_squares(curve_err,n0,bounds=[-40,40],args=(values,x))
        return res['x']


# find curve model for partials
def find_curved_partial(partial,ts,display = 0):
    if display == 1:
        ax_partial_curve = plt.axes(projection="3d")

    partial_n =[]
    partial_curve = np.zeros(partial.shape)
    pt_n = []

    for i in range(partial.shape[0]):
        pt = partial[i,:]

        pt_sm = smooth(pt)
        points = find_adsr(pt_sm)
        points.insert(0,0)
        points.append(pt_sm.shape[0])

        for j in range(5):
            st_ind = points[j]
            ed_ind = points[j+1]
            if st_ind >= ed_ind:
                pt_n.append(1)
            else:
                n = find_opt_n(pt_sm[st_ind:ed_ind,np.newaxis],ts[st_ind:ed_ind])
                pt_n.append(n)
                partial_curve[i,st_ind:ed_ind] = curve(n,pt_sm[st_ind:ed_ind,np.newaxis],ts[st_ind:ed_ind])[:,0]

        partial_n.append(pt_n)
        pt_n = []

        # plotting
        if display == 1:
            ax_partial_curve.plot(ts[:,0],freq_peak[i]*np.ones((ts.shape[0],1))[:,0],partial_curve[i,:].T)
            ax_partial_curve.set_ylim([0,4000])
    
    if display == 1:
        plt.show()
    
    return partial_n,partial_curve

## find spectrum from partials
#def partial2spectrum(partial,freq_peaks,freqs,spec_size):
#    spectrum = np.zeros(spec_size)
#    for i in range(freq_peaks):
#        
#    
#    return spectrum

if __name__ == '__main__':
    # file_path = 'sound/diziB4.wav'
    # file_path = 'sound/erhuB4.wav'
    # file_path = 'sound/fluteB4.wav'
    # file_path = 'sound/violinB4.wav'

    file_path = '../../sounds/flute-A4.wav'
    
    sd_data, nchannels, sampwidth, framerate, nframes = get_sound_data(file_path)
#    play_sound(sd_data, nchannels, sampwidth, framerate)
#    plot_wave(sd_data, framerate, nframes)
    spectrum,phase,freqs,ts = find_spectrum(sd_data,framerate)
    freq_peak, partial = find_partial(spectrum,freqs)
    partial_n, partial_curve = find_curved_partial(partial,ts,1)
    print(partial_n)

#
##import matplotlib
##matplotlib.use('TkAgg')
#
#import matplotlib.pyplot as plt
#
#import numpy as np
#fig = plt.figure()
#plt.plot(np.arange(5))
#fig.show()

