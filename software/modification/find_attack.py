import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../transformations/'))
sys.path.append('/Library/Python/2.7/site-packages/')

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.interpolate as itpl

import utilFunctions as UF
import hpsModel as HPS
import harmonicModel as HM
import utilModi as UM
import timbreModi as TM

def find_attack_bspline(x):
    # scipy.signal.bspline
    y = sig.bspline(x,3)
    return y

def find_attack_univariateSpline(x,lam,k=3):
    # scipy.interpolation.univarspl
    t = np.arange(x.size)
    spl = itpl.UnivariateSpline(t,x,k=k,s=lam)
    # ts = np.linspace(0,x.size,30000)

    y = spl(t)

    # test
    dy_spl=spl.derivative()

    return t,y,spl,dy_spl

def find_attack_derivative_tracking(x,lam):
    # detect attack by derivative tracking scheme
    t, y, y_spl, dy_spl = find_attack_univariateSpline(x,lam,k=4)
    dy = dy_spl(t)
    dyRoot = dy_spl.roots()
    print(dyRoot)

def find_spline_harmonic(x,lam): # do spline smoothing to a harmonic
    t = np.arange(x.size)
    spl = itpl.UnivariateSpline(t, x)
    spl.set_smoothing_factor(lam)
    return spl(t)

def frame_energy(X): # calculate energy for transient detection
    N = X.size
    energy = np.sum(X[1:N]**2)
    return energy

def frame_high_frequency_content(X): # calculate energy for transient detection
    N = X.size
    ind = np.arange(2,N+1)
    HFC = np.sum((X[1:N]**2) * ind)
    return HFC

def find_transient_level(mX):
    frameNum, binNum = mX.shape
    transLevel = np.empty(0)
    prev_HFC = frame_high_frequency_content(mX[0,:])
    for i in range(1,frameNum):
        cur_energy = frame_energy(mX[i,:])
        cur_HFC = frame_high_frequency_content(mX[i,:])
        transLevel = np.append(transLevel,(cur_HFC**2)/prev_HFC/cur_energy)
        prev_HFC = cur_HFC

    return(transLevel)


if __name__ == "__main__":

    exp = 3

    if exp == 0:
        x = np.linspace(-3, 3, 50)
        y = np.exp(-x ** 2) + 0.1 * np.random.randn(50)
        plt.plot(x, y, 'ro', ms=5)

        spl = itpl.UnivariateSpline(x, y)
        xs = np.linspace(-3, 3, 1000)
        plt.plot(xs, spl(xs), 'g', lw=3)
        spl.set_smoothing_factor(0.5)
        plt.plot(xs, spl(xs), 'b', lw=3)
        plt.show()


    elif exp == 1: # test find_attack_bspline
        # file_path = '../../sounds/flute_acoustic/flute_acoustic_valid/flute_acoustic_002-084-050.wav'
        # file_path = '../../sounds/brass_acoustic/brass_acoustic_valid/brass_acoustic_006-069-075.wav'
        # file_path = '../../sounds/string_acoustic/string_acoustic_valid/string_acoustic_012-048-050.wav'
        file_path = '../../sounds/reed_acoustic/reed_acoustic_valid/reed_acoustic_037-069-050.wav'

        nH = 1
        minf0 = UM.pitch2freq(69) - 50
        maxf0 = UM.pitch2freq(69) + 50

        x, fs, hfreq, hmag, hphase, stocEnv = TM.hps_ana(file_path, nH=nH, minf0=minf0, maxf0=maxf0, N=8192, M=8192)

        for i in range(nH):
            harm = hmag[:,i]
            harmZero = -500 * (harm == 0)
            harm = harm + harmZero

            # linear smoothing
            smLength = 11
            harm = UM.smooth(harm, smLength)

            # scipy.signal.bspline
            # harm_bspl = find_attack_bspline(harm)

            # scipy.interpolation.univarspl
            t,y,y_spl,dy_spl= find_attack_univariateSpline(harm,lam = 5e3)

            # plotting
            plt.subplot(2,1,1)
            plt.plot(np.arange(harm.size)*128/fs,harm)
            plt.plot(t*128/fs,y)
            plt.subplot(2, 1, 2)
            plt.plot(dy_spl(t))
            plt.show()

            # try derivative tracking
            # find_attack_derivative_tracking(harm,lam=1e4)


    elif exp == 2: # test splined sound
        # file_path = '../../sounds/flute_acoustic/flute_acoustic_valid/flute_acoustic_002-084-050.wav'
        # file_path = '../../sounds/brass_acoustic/brass_acoustic_valid/brass_acoustic_006-069-075.wav'
        # file_path = '../../sounds/string_acoustic/string_acoustic_valid/string_acoustic_012-048-050.wav'
        file_path = '../../sounds/reed_acoustic/reed_acoustic_valid/reed_acoustic_037-069-050.wav'

        nH = 20
        minf0 = UM.pitch2freq(69) - 50
        maxf0 = UM.pitch2freq(69) + 50

        x, fs, hfreq, hmag, hphase, stocEnv = TM.hps_ana(file_path, nH=nH, minf0=minf0, maxf0=maxf0, N=8192, M=8192)

        hfreqSyn = hfreq.copy()
        hmagSyn = hmag.copy()
        hphaseSyn = hphase.copy()

        for i in range(nH):
            harm = hmag[:,i]

            # linear smoothing
            smLength = 11
            harm = UM.smooth(harm, smLength)

            # scipy.interpolation.univariatespline
            hmagSyn[:,i] = find_spline_harmonic(harm,lam = 1e4)


        # synthesize sound
        # FFT SIZE FOR SYNTHESIS
        Ns = 512

        # HOP SIZE
        H = 128

        # synthesis

        # hfreqSyn = np.repeat(hfreqSyn[0:1, :], hfreqSyn.shape[0], axis=0)
        y, yh, yst = HPS.hpsModelSynth(hfreqSyn, hmagSyn, hphaseSyn, stocEnv, Ns, H, fs)

        outputFileSines = 'output_sounds/syn_sines.wav'
        outputFileStochastic = 'output_sounds/syn_stochastic.wav'
        outputFile = 'output_sounds/syn.wav'
        UF.wavwrite(yh, fs, outputFileSines)
        UF.wavwrite(yst, fs, outputFileStochastic)
        UF.wavwrite(y, fs, outputFile)

        # play sound
        UF.wavplay(file_path)
        UF.wavplay(outputFile)

        # display harmonics
        t = np.arange(hfreq.shape[0]) * x.shape[0] / fs / hfreq.shape[0]
        plt.figure()
        UM.plot_spec3d(hfreq, hmag, t, 1)
        plt.figure()
        UM.plot_spec3d(hfreqSyn, hmagSyn, t, 1)

    elif exp == 3: # transient detection test
        # instruments = ['flute', 'oboe', 'saxophone', 'horn', 'violin', 'erhu', 'guitar', 'harp', 'piano', 'organ', 'trumpet']
        file_path = '../../sounds/A4/trumpet-A4.wav'

        # plot spectrogram
        plt.figure()
        mX,pX,fs,H = UM.plot_spectrogram(file_path)
        # print(mX.shape) # (142,2049) when N = 4096

        # find transient level
        transLevel = find_transient_level(UM.dB2abslt(mX))
        print(transLevel[0:40])

        # plotting
        t = np.arange(transLevel.size)*H/fs
        plt.plot(t,transLevel)
        plt.show()






    # ###### test frame_energy(X) ##############
    # X = np.arange(6)
    # print(frame_energy(X))
    # #########################################

    # ###### test frame_high_frequency_content(X) ##############
    # X = np.arange(6)
    # print(frame_high_frequency_content(X))
    # # #########################################

