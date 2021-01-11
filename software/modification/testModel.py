# functions that test validation of sms-tool functions (models and transformations) and pyAudioAnalysis descriptors

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../transformations/'))
sys.path.append('/Library/Python/2.7/site-packages/')

import matplotlib
matplotlib.use('MacOSX')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import get_window,find_peaks
from pyAudioAnalysis import audioAnalysis as aa

import stft
import utilFunctions as UF
import hpsModel as HPS
import hpsTransformations as HPST
import harmonicTransformations as HT


def play_sd():
    file_path = '../../sounds/pianoStrings-A4.wav'
    UF.wavplay(file_path)


def try_ana_syn():
    file_path = '../../sounds/pianoStrings-A4.wav'

    # define window type
    window='blackman'

    # window size M
    M=601

    # fft size
    N=2048

    # FFT SIZE FOR SYNTHESIS
    Ns = 512

    # HOP SIZE
    H = 128

    # threshold for harmonics in dB
    t= -100

    # min sinusoid duration to be considered as harmonics
    minSineDur=0.1

    #MAX NUMBER OF HARMONICS
    nH=100

    #MIN FUNDAMENTAL FREQUENCY
    minf0=350

    #MAX FUNDAMENTAL FREQUENCY
    maxf0=700

    #MAX ERROR ACCEPTED IN F0 DETECTION
    f0et=5

    #MAX ALLOWED DEVIATION OF HARMONIC TRACKS
    harmDevSlope=0.01

    #DECIMATION FACTOR FOR STOCHASTIC GENERATION
    stocf=0.1

    # read input sound

    (fs, x) = UF.wavread(file_path)

    # compute the harmonic plus stochastic model of the whole sound
    # analysis
    w = get_window(window, M)
    # y, yh, yst = HPS.hpsModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, stocf)
    hfreq, hmag, hphase, stocEnv = HPS.hpsModelAnal(x,fs,w,N,H,t,nH,minf0,maxf0,f0et,harmDevSlope,minSineDur,Ns,stocf)

    # synthesis
    y, yh, yst = HPS.hpsModelSynth(hfreq, hmag, hphase, stocEnv, Ns, H, fs)

    # output sound file (monophonic with sampling rate of 44100)
    outputFileSines = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_sines.wav'
    outputFileStochastic = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_stochastic.wav'
    outputFile = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel.wav'

    # write sounds files for harmonics, stochastic, and the sum
    UF.wavwrite(yh, fs, outputFileSines)
    UF.wavwrite(yst, fs, outputFileStochastic)
    UF.wavwrite(y, fs, outputFile)

    # play output sound
    UF.wavplay('output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_sines.wav')
    UF.wavplay('output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_stochastic.wav')
    UF.wavplay('output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel.wav')

def try_dscpt():

    # read input sound
    file_path = '../../sounds/flute-A4.wav'
    (fs, x) = UF.wavread(file_path)

    ## parameters
    # # define window type
    window='blackman'

    # window size M
    M=2048

    # fft size
    N=4096

    # FFT SIZE FOR SYNTHESIS
    Ns = 512

    # HOP SIZE
    H = 128

    ## testing descriptor extraction
    # spec, ts, freqs = aa.sF.spectrogram(x,fs,N,M)  # M no less than 1/2 of N

    # chrom, ts_ch, freqs_ch = aa.sF.chromagram(x,fs,N,M,1)

    fts, fts_name = aa.sF.feature_extraction(x,fs,N,M,1)
    print(fts.shape,len(fts_name))
    for i in range(len(fts_name)):
        print(fts_name[i],fts[i])



def try_trans():

    file_path = '../../sounds/flute-A4.wav'
    (fs, x) = UF.wavread(file_path)

    ## parameters
    # define window type
    window = 'blackman'

    # window size M
    M = 2048

    # fft size
    N = 4096

    # FFT SIZE FOR SYNTHESIS
    Ns = 512

    # HOP SIZE
    H = 128

    # threshold for harmonics in dB
    t = -100

    # min sinusoid duration to be considered as harmonics
    minSineDur = 0.1

    # MAX NUMBER OF HARMONICS
    nH = 20

    # MIN FUNDAMENTAL FREQUENCY
    minf0 = 350

    # MAX FUNDAMENTAL FREQUENCY
    maxf0 = 700

    # MAX ERROR ACCEPTED IN F0 DETECTION
    f0et = 5

    # MAX ALLOWED DEVIATION OF HARMONIC TRACKS
    harmDevSlope = 0.01

    # DECIMATION FACTOR FOR STOCHASTIC GENERATION
    stocf = 0.1

    # analysis
    w = get_window(window, M)
    hfreq, hmag, hphase, mYst = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur,
                                                 Ns, stocf)

    # transformation(scaling) parameters
    freqScaling = np.array([0, 1.2, 2.01, 1.2, 2.679, .7, 3.146, .7])
    freqStretching = np.array([0, 1, 2.01, 1, 2.679, 1.5, 3.146, 1.5])
    timeScaling = np.array([0, 0, 2.138, 2.138 - 1.0, 3.146, 3.146])
    timbrePreservation = 1


def find_spectrum(sd_data, framerate, display=0):
    window_size = 2048
    window = get_window('blackman', window_size)
    FFT_size = 4096
    Hop_size = 512

    mX, pX = stft.stftAnal(sd_data, window, FFT_size, Hop_size)  # get amplitude spectrum and phase spectrum
    spectrum = mX.transpose()
    phase = pX.transpose()

    freq_frm = spectrum.shape[0]
    freqs = np.arange(freq_frm) * (float(framerate) / FFT_size)
    freqs = freqs[:, np.newaxis]

    time_frm = spectrum.shape[1]
    ts = (np.arange(time_frm) + 0.5) * (Hop_size / float(framerate))
    ts = ts[:, np.newaxis]

    if display == 1:
        fig1 = plt.figure()
        plt.subplot(2, 1, 1)
        plt.pcolormesh(ts[:, 0], freqs[:, 0], spectrum, cmap=cm.coolwarm)
        plt.xlabel('time (sec)')
        plt.ylabel('frequency (Hz)')
        plt.title('magnitude spectrogram')

        plt.subplot(2, 1, 2)
        plt.pcolormesh(ts[:, 0], freqs[:, 0], phase, cmap=cm.coolwarm)
        plt.xlabel('time (sec)')
        plt.ylabel('frequency (Hz)')
        plt.title('phase spectrogram (derivative)')

        plt.show()

    # 3D display
    fig2 = plt.figure()
    if display == 2:
        y = np.repeat(freqs, ts.shape[0], axis=1)
        x = np.repeat(ts, freqs.shape[0], axis=1).T
        z = spectrum
        ax = plt.axes(projection="3d")
        ax.plot_surface(x, y, z, cmap=cm.coolwarm)
        ax.axis()
        plt.show()

    return spectrum, phase, freqs, ts


def find_partial(spectrum, freqs, ts, display=0):
    freq_sum = np.sum(spectrum, axis=1)
    freq_peak_ind = find_peaks(freq_sum, height=-100)
    freq_peak_ind = freq_peak_ind[0]
    freq_peak = freqs[freq_peak_ind]

    # find amplitude of each partial
    partial = []
    for i in freq_peak_ind:
        amp = spectrum[i, :]
        partial.append(amp)

    partial = np.array(partial)
    # N_pt = partial.shape[0]

    ## plot partials
    if display == 1:
        ax_partial = plt.axes(projection="3d")
        fig = plt.figure()
        for i in range(partial.shape[0]):
            ax_partial.plot(ts[:, 0], freq_peak[i] * np.ones((ts.shape[0], 1))[:, 0], partial[i, :].T, )
            ax_partial.set_ylim([0, 4000])
        plt.show()
    return freq_peak, partial

if __name__ == "__main__":

    play_sd()

    # try_ana_syn()
    # try_dscpt()
    # try_trans()
    #
    file_path = '../../sounds/flute-A4.wav'
    (fs, x) = UF.wavread(file_path)
    spectrum, phase, freqs, ts = find_spectrum(x, fs, 2)
    freq_peak, partial = find_partial(spectrum, freqs, ts, 1)
    print(freqs, '\n')
    print(freq_peak)









    