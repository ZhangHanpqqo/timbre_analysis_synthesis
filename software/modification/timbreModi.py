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

import utilFunctions as UF
import hpsModel as HPS
import harmonicModel as HM
import utilModi as UM
import find_attack as FA


def spec_modi(file_path, showTest, modiType, flatSpec):
    (fs, x) = UF.wavread(file_path)

    # # parameters
    # # window shapex
    # window = 'blackmanharris'

    # # window size M
    # M = 4096

    # # fft size
    # N = 8192

    # # FFT SIZE FOR SYNTHESIS
    # Ns = 512

    # # HOP SIZE
    # H = 128

    # # threshold for harmonics in dB
    t = -200

    # min sinusoid duration to be considered as harmonics
    minSineDur = 0.1

    # MAX NUMBER OF HARMONICS
    nH = 40

    # MIN FUNDAMENTAL FREQUENCY
    minf0 = 400

    # MAX FUNDAMENTAL FREQUENCY
    maxf0 = 500

    # MAX ERROR ACCEPTED IN F0 DETECTION
    f0et = 5

    # MAX ALLOWED DEVIATION OF HARMONIC TRACKS
    harmDevSlope = 0.01

    # DECIMATION FACTOR FOR STOCHASTIC GENERATION
    stocf = 0.1

    ######## trail 1: based on spectrum ########
    # get some features of the initial audio
    # get spectrum
    # fts, ftsName = aa.sF.feature_extraction(x, fs, N, M, 1)
    # specCentroidOrg = fts[3,::]
    # specEnergyOrg = fts[1,::]
    #
    # if showTest == 1:
    #     print('spectrum centroid\n',specCentroidOrg,'\n energy \n',specEnergyOrg)

    # construct a bandpass filter

    ######### train 1 end #############

    ######## trail 2: based on harmonics ########
    # get some features of the initial audio
    fts, ftsName = aa.sF.feature_extraction(x, fs, N, M, 1)
    specCentroidOrg = fts[3,::]
    specCentroidOrgMean = np.mean(specCentroidOrg)
    specCentroidOrgVar = np.var(specCentroidOrg)
    specEnergyOrg = fts[1,::]
    specEnergyOrgMean = np.mean(specEnergyOrg)
    specEnergyOrgVar = np.var(specEnergyOrg)
    specFluxOrg = fts[6,::]
    specFluxOrgMean = np.mean(specFluxOrg)
    specFluxOrgVar = np.var(specFluxOrg)

    if showTest >= 1:
        #print('spectrum centroid\n',specCentroidOrg,'\n energy \n',specEnergyOrg)
        print('spectrum centroid mean\n', specCentroidOrgMean, '\n variance \n', specCentroidOrgVar)
        print('spectrum energy mean\n', specEnergyOrgMean, '\n variance \n', specEnergyOrgVar)
        print('spectrum flux mean\n', specFluxOrgMean, '\n variance \n', specFluxOrgVar)

    # find harmonics
    w = sig.get_window(window, M)
    hfreq, hmag, hphase, stocEnv = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur,
                                                    Ns, stocf)

    if flatSpec == 1:
        hmag = -50*np.ones(hmag.shape)
        hphase = np.zeros(hphase.shape)

    if showTest == 1:
        print('hfreq\n',hfreq, '\n harmonics magnitude\n',hmag,'\n harmonics phase\n',hphase,'\n stochastic\n',stocEnv)

    # synthesize the original sound
    y, yh, yst = HPS.hpsModelSynth(hfreq, hmag, hphase, stocEnv, Ns, H, fs)

    # output sound file (monophonic with sampling rate of 44100)
    outputFileSines = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_sines.wav'
    outputFileStochastic = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_stochastic.wav'
    outputFile = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel.wav'

    # write sounds files for harmonics, stochastic, and the sum
    UF.wavwrite(yh, fs, outputFileSines)
    UF.wavwrite(yst, fs, outputFileStochastic)
    UF.wavwrite(y, fs, outputFile)

    if modiType == 'EH':
        # try even harmony attenuation
        attenRatio = 0.3
        hmagModi = UM.evenAtten(hmag, attenRatio)
    if modiType == 'OH':
        # try odd harmony attenuation
        attenRatio = 0.3
        keepF0 = 1
        hmagModi = UM.oddAtten(hmag, attenRatio, keepF0)
    elif modiType == 'exp':
        rate = 0.4
        hmagModi = UM.expAtten(hmag,rate)
    elif modiType == 'gaussian':
        freqMean = 3000
        freqVar = 2000
        keepF0 = 0
        hmagModi = UM.gaussianModi(hmag, hfreq, freqMean, freqVar,keepF0)
    elif modiType == 'invGaussian':
        freqMean = 6000
        freqVar = 2000
        keepF0 = 0
        hmagModi = UM.invGaussianModi(hmag, hfreq, freqMean, freqVar,keepF0)
    elif modiType == 'single':
        harmNo = 3
        rate = 0.6
        hmagModi = UM.singleModi(hmag, harmNo, rate)

    # synthesize the attenuated sound
    yModi, yhModi, ystModi = HPS.hpsModelSynth(hfreq, hmagModi, hphase, stocEnv, Ns, H, fs)

    if modiType == 'EH':
        outputFileSines = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_EHAttened_'+str(attenRatio)+'_sines.wav'
        outputFileStochastic = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_EHAttened_'+str(attenRatio)+'_stochastic.wav'
        outputFile = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_EHAttened_'+str(attenRatio)+'.wav'
    if modiType == 'OH':
        outputFileSines = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_OHAttened_'+str(attenRatio)+'_sines.wav'
        outputFileStochastic = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_OHAttened_'+str(attenRatio)+'_stochastic.wav'
        outputFile = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_OHAttened_'+str(attenRatio)+'.wav'
    elif modiType == 'exp':
        outputFileSines = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_expAttened_' + str(rate) + '_sines.wav'
        outputFileStochastic = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_expAttened_' + str(rate) + '_stochastic.wav'
        outputFile = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_expAttened_' + str(rate) + '.wav'
    elif modiType == 'gaussian':
        outputFileSines = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_gaussian_m' + str(freqMean)+'_v'+str(freqVar) + '_sines.wav'
        outputFileStochastic = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_gaussian_m' + str(freqMean)+'_v'+str(freqVar) + '_stochastic.wav'
        outputFile = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_gaussian_m' + str(freqMean)+'_v'+str(freqVar) + '.wav'
    elif modiType == 'invGaussian':
        outputFileSines = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_invgaussian_m' + str(freqMean) + '_v' + str(freqVar) + '_sines.wav'
        outputFileStochastic = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_invgaussian_m' + str(freqMean) + '_v' + str(freqVar) + '_stochastic.wav'
        outputFile = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_invgaussian_m' + str(freqMean) + '_v' + str(freqVar) + '.wav'
    elif modiType == 'single':
        outputFileSines = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_single_hno' + str(harmNo)+'_r'+str(rate) + '_sines.wav'
        outputFileStochastic = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_gaussian_hno' + str(harmNo)+'_r'+str(rate) + '_stochastic.wav'
        outputFile = 'output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_gaussian_hno' + str(harmNo)+'_r'+str(rate) + '.wav'


    # write synthesized audio in .wav
    UF.wavwrite(yhModi, fs, outputFileSines)
    UF.wavwrite(ystModi, fs, outputFileStochastic)
    UF.wavwrite(yModi, fs, outputFile)

    # play original sound
    if showTest == 1:
        # UF.wavplay(file_path)
        UF.wavplay('output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_sines.wav')

        # play synthesized attenuated sound
        UF.wavplay(outputFileSines)

    # get features from the modified audio
    ftsModi, ftsNameModi = aa.sF.feature_extraction(yModi, fs, N, M, 1)
    specCentroidModi = ftsModi[3, ::]
    specCentroidModiMean = np.mean(specCentroidModi)
    specCentroidModiVar = np.var(specCentroidModi)
    specEnergyModi = ftsModi[1, ::]
    specEnergyModiMean = np.mean(specEnergyModi)
    specEnergyModiVar = np.var(specEnergyModi)
    specFluxModi = ftsModi[6, ::]
    specFluxModiMean = np.mean(specFluxModi)
    specFluxModiVar = np.var(specFluxModi)

    if showTest >= 1:
        #print('spectrum centroid\n',specCentroidOrg,'\n energy \n',specEnergyOrg)
        print('modified spectrum centroid mean\n', specCentroidModiMean, '\n variance \n', specCentroidModiVar)
        print('modified spectrum energy mean\n', specEnergyModiMean, '\n variance \n', specEnergyModiVar)
        print('modified spectrum flux mean\n', specFluxModiMean, '\n variance \n', specFluxModiVar)

    # plotting
    if showTest == 1:
        midInd = round(hfreq.shape[0]/2)
        hfreqFl = hfreq[midInd,::]
        hmagFl = hmag[midInd,::]
        hmagModiFl = hmagModi[midInd,::]
        # plot harmonics
        plt.figure(1)
        plt.plot(hfreqFl,hmagFl,'k*-')
        plt.plot(hfreqFl,hmagModiFl,'b.-')
        plt.show()

    ######### train 2 end #############

    print(specCentroidOrgMean, specCentroidOrgMean*50.1136, specFluxOrgMean, specEnergyOrgMean)

def hps_ana(file_path, nH=40, minf0=100, maxf0=1000, M=4001,N=8192,Ns=512,H=128):
    (fs, x) = UF.wavread(file_path)

    # parameters
    # window shapex
    window = 'blackmanharris'

    # window size M
    # M = 4001

    # fft size
    # N = 8192

    # FFT SIZE FOR SYNTHESIS
    # Ns = 512

    # HOP SIZE
    # H = 128

    # threshold for harmonics in dB
    t = -200

    # min sinusoid duration to be considered as harmonics
    minSineDur = 0.1

    # MAX NUMBER OF HARMONICS
    # nH = 40

    # MIN FUNDAMENTAL FREQUENCY
    # minf0 = 400

    # MAX FUNDAMENTAL FREQUENCY
    # maxf0 = 500

    # MAX ERROR ACCEPTED IN F0 DETECTION
    f0et = 5

    # MAX ALLOWED DEVIATION OF HARMONIC TRACKS
    harmDevSlope = 0.01

    # DECIMATION FACTOR FOR STOCHASTIC GENERATION
    stocf = 0.1

    # dbg
    # print('Hop size in hps_ana',H)


    w = sig.get_window(window, M)
    hfreq, hmag, hphase, stocEnv = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur,
                                                    Ns, stocf)

    # freqMean = np.mean(hfreq,axis=0)
    # freqVar = np.var(hfreq,axis=0)

    return x, fs, hfreq, hmag, hphase, stocEnv

def spec_synth(x, fs, hfreq, hmag, hphase, stocEnv, showTest, mode, minf0=100, maxf0=1000,select=0,trim=1):

    # f0 = 440 # hardcode
    # parameters
    # window shapex
    window = 'blackmanharris'

    # window size M
    M = 4096*2

    # fft size
    N = 4096*2

    # FFT SIZE FOR SYNTHESIS
    Ns = 512

    # HOP SIZE
    H = 128

    # threshold for harmonics in dB
    t = -200

    # min sinusoid duration to be considered as harmonics
    minSineDur = 0.1

    # MAX NUMBER OF HARMONICS
    nH = 20

    # MIN FUNDAMENTAL FREQUENCY
    # minf0 = 400

    # MAX FUNDAMENTAL FREQUENCY
    # maxf0 = 500

    # MAX ERROR ACCEPTED IN F0 DETECTION
    f0et = 5

    # MAX ALLOWED DEVIATION OF HARMONIC TRACKS
    harmDevSlope = 0.01

    # DECIMATION FACTOR FOR STOCHASTIC GENERATION
    stocf = 0.1

    # tst
    # hfreq = hfreq[:,0:1]
    # hmag = hmag[:,0:1]
    # hphase = hphase[:,0:1]

    # initialization
    hfreqSyn = hfreq.copy()
    hmagSyn = hmag.copy()
    hphaseSyn = hphase.copy()
    hfreqIntp = hfreq.copy()
    hmagIntp = hmag.copy()

    if trim in [1]:
        hfreqNoise = UM.find_freq_failure(hfreq,minf0,maxf0)
        hfreqIntp, hmagIntp, hphaseIntp = UM.mag_interpolate(hfreq, hfreqNoise, hmag, hphase)
        hfreqSyn = hfreqIntp.copy()
        hmagSyn = hmagIntp.copy()
        hphaseSyn = hphaseIntp.copy()


    if mode in [1,4,6,7]:  #
        w = sig.get_window(window, M)

        # frequency generation
        f0= np.mean(HM.f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et))
        # f0 = 440
        fIntv = f0
        fFund = f0

        freqMean, freqVar = UM.freq_ana(hfreq,f0,0)
        hfreqBase = np.arange(hfreq.shape[1]) * fIntv + fFund
        hfreqBase = np.array([hfreqBase])
        hfreqBase = np.repeat(hfreqBase, hfreq.shape[0], axis = 0)

        varRate = 0
        hfreqDist = []

        smFreq = 31 # choose the box size for smoothing. Need to be odd. If not to smooth, set 1.
        for i in range(hfreq.shape[1]):

            # mu = freqMean[i] - (i+1)*f0
            mu = freqMean[i]
            sigma = freqVar[i] * varRate

            s = np.random.normal(mu,sigma,hfreq.shape[0])
            s = UM.smooth(s,smFreq)
            hfreqDist.append(s)


        hfreqDist = np.array(hfreqDist).T
        hfreqSyn = hfreqBase+hfreqDist

    t = np.arange(hfreq.shape[0]) * x.shape[0] / fs / hfreq.shape[0]
    tArray = np.array([t]).T
    if mode in [2,4,5,7]:
        ########## harmonics synthesis with curve model #############
        hns, hCurve, hPoints = UM.find_curved_partial(hmagIntp.T, tArray)
        hmagSyn = hCurve.T

        hADSREpoch = (tArray[:,0])[hPoints]
        hADSRPortion = hPoints/hmag.shape[0]
        aTime = hADSREpoch[:,1]-hADSREpoch[:,0]
        dsTime = hADSREpoch[:,2]-hADSREpoch[:,1]
        rTime = hADSREpoch[:,3]-hADSREpoch[:,2]


        ## tst
        # print(hns)
        # UM.save_matrix('../../../test.xlsx',hns,sheetName='erhuCurveN')
        # print('Epoch:\n', hADSREpoch,'\n Attach:\n', np.array([aTime]).T,'\n Develop+Sustain:\n',np.array([dsTime]).T,'\n Release:\n', np.array([rTime]).T,'\n')

        ########## harmonics using shape of the fundamental ########
        # w = hmag[0,0]-hmag[::,0]
        # hmagSyn = np.repeat(np.array([hmag[0,::]]),hmag.shape[0],axis=0)
        # wh = np.repeat(np.array([w]).T,hmag.shape[1],axis = 1)
        # hmagSyn = hmagSyn - wh

        ################# add harmonics ###########################
        # f = 200
        # v = -50
        # freqAdd = np.ones((hfreq.shape[0],1)) * f
        # valueAdd = np.ones((hmag.shape[0],1)) * v
        # phaseAdd = np.ones((hphase.shape[0],1)) * 0

        # freqAdd = hfreq[::,0::1]*0.5
        # valueAdd = hmag[::,0::1]
        # phaseAdd = hphase[::,0::1]
        #
        # hfreqSyn, hmagSyn, hphaseSyn = UM.add_harm(hfreqSyn, hmagSyn, hphaseSyn, freqAdd, valueAdd, phaseAdd)

    if mode in [3,5,6,7]:
        # get the phase of the first frame randomly
        # hphaseSyn[0,:] = 2*np.pi*np.random.rand(hphaseSyn[0,:].size)

        # use the original first frame
        # hphaseSyn[0, :] = hphase[0, :]

        # try to assign the first frame phase linearly
        ffphase = hphase[0,:]
        ffphaseSyn, ffpSlope, ffpIntercept, ffpErr = UM.find_first_frame_phase(ffphase, mode = 'linear')
        hphaseSyn[0,:] = ffphaseSyn

        ## tst
        # print("first frame phase:")
        # print(ffpSlope,ffpIntercept,ffpErr)

        # propagate to the following frames, according to frequency
        for l in range(1,hphaseSyn.shape[0]):
            hphaseSyn[l,:] = hphaseSyn[l-1,:] + (np.pi*(hfreqSyn[l-1,:]+hfreqSyn[l,:])/fs)*H

        hphaseSyn = np.unwrap(hphaseSyn, axis=0)


    ########### select some harmonics #######################
    if select in [1]:
        nh = hfreq.shape[1]
        nf = hfreq.shape[0]
        # slctHarm = np.array([1])  # select some harmonics
        # slctHarm = np.arange(7, nH)
        slctHarm = np.arange(0, 10)
        slctHarm = slctHarm[slctHarm<nh]

        # select synthesized spectrum
        hfreqSlct = np.zeros([nf,1])    # initialization
        hmagSlct = np.zeros([nf,1])
        hphaseSlct = np.zeros([nf,1])
        for i in range(slctHarm.shape[0]):
            hNo = slctHarm[i]
            freqAdd = hfreqSyn[::,hNo:(hNo+1)]
            valueAdd = hmagSyn[::,hNo:(hNo+1)]
            phaseAdd = hphaseSyn[::,hNo:(hNo+1)]
            hfreqSlct, hmagSlct, hphaseSlct = UM.add_harm(hfreqSlct, hmagSlct, hphaseSlct, freqAdd, valueAdd, phaseAdd)

        hfreqSyn = hfreqSlct[::,1:]
        hmagSyn = hmagSlct[::, 1:]
        hphaseSyn = hphaseSlct[::, 1:]

        # select original spectrum
        hfreqSlct = np.zeros([nf, 1])  # initialization
        hmagSlct = np.zeros([nf, 1])
        hphaseSlct = np.zeros([nf, 1])
        for i in range(slctHarm.shape[0]):
            hNo = slctHarm[i]
            freqAdd = hfreq[::, hNo:(hNo + 1)]
            valueAdd = hmag[::, hNo:(hNo + 1)]
            phaseAdd = hphase[::, hNo:(hNo + 1)]
            hfreqSlct, hmagSlct, hphaseSlct = UM.add_harm(hfreqSlct, hmagSlct, hphaseSlct, freqAdd, valueAdd, phaseAdd)

        hfreq = hfreqSlct[::, 1:]
        hmag = hmagSlct[::, 1:]
        hphase = hphaseSlct[::, 1:]
    ########################################################


    # hSpec = fft(hmag[::,9])
    # print(hSpec)
    # plt.plot(hSpec)
    # plt.show()


    # FFT SIZE FOR SYNTHESIS
    Ns = 512

    # HOP SIZE
    H = 128

    # synthesis

    # hfreqSyn = np.repeat(hfreqSyn[0:1, :], hfreqSyn.shape[0], axis=0)
    y, yh, yst = HPS.hpsModelSynth(hfreqSyn, hmagSyn, hphaseSyn, stocEnv, Ns, H, fs)

    ## tst
    # print(hfreqSyn[:,0],'\n',hmagSyn[:,0],'\n',hphaseSyn[:,0])
    # plt.plot(yh)
    # plt.show()
    # y, yh, yst = HPS.hpsModelSynth(hfreqSyn, hmag, hphaseSyn, stocEnv, Ns, H, fs)

    # output file
    outputFileSines = 'output_sounds/syn_sines.wav'
    outputFileStochastic = 'output_sounds/syn_stochastic.wav'
    outputFile = 'output_sounds/syn.wav'

    # write synthesized audio in .wav
    UF.wavwrite(yh, fs, outputFileSines)
    UF.wavwrite(yst, fs, outputFileStochastic)
    UF.wavwrite(y, fs, outputFile)

    if showTest in [1,3]:
        UF.wavplay(file_path)
        # UF.wavplay('output_sounds/' + os.path.basename(file_path)[:-4] + '_hpsModel_sines.wav')
        UF.wavplay(outputFile)
    if showTest in [2,3]:
        plt.figure()
        UM.plot_spec3d(hfreq, hmag, t,1)
        # plt.figure()
        # UM.plot_spec3d(hfreqIntp, hmagIntp, t)
        plt.figure()
        UM.plot_spec3d(hfreqSyn, hmagSyn, t,1)
        # UM.plot_spec3d_cmp(hfreq, hmag, t, hfreqSyn, hmagSyn, t)

        # plt.figure()
        # UM.plot_spectrogram(outputFile)

    if showTest in [4]:
        plt.figure()
        UM.plot_spec3d(hfreq, hphase, t,2)
        plt.figure()
        UM.plot_spec3d(hfreq, hphaseSyn, t,2)
        plt.figure()
        UM.plot_spec3d(hfreq, np.abs(hphaseSyn-hphase)%(2*np.pi), t,2)

    return hfreqSyn, hmagSyn, hphaseSyn

def sound_info_clct(file_path, sdInfo, nH = 40, minf0 = 100, maxf0 = 1000, M=4001, N=8192, Ns=512, H=128):
    x, fs, hfreq, hmag, hphase, stocEnv = hps_ana(file_path, nH=nH, minf0=minf0, maxf0=maxf0, M=M, N=N, Ns=Ns, H=H)
    sdInfo['fs']=fs
    sdInfo['stocEnv']=stocEnv

    # save_path = 'result/test.xlsx'
    # UM.save_matrix(save_path, hfreq, sheetName='guitar_freq_original')

    # delete frames at the beginning where no harmonics are detected, i.e. silence
    nonSlc = UM.non_silence_dtct(hfreq[:, 0], minf0, maxf0)
    hfreq = hfreq[nonSlc[0]:nonSlc[1]+1,:]
    hmag = hmag[nonSlc[0]:nonSlc[1] + 1, :]
    hphase = hphase[nonSlc[0]:nonSlc[1] + 1, :]

    # do interpolation, where harmonics are not detected within the sound(freq out of [minf0,maxf0])
    hfreqNoise = UM.find_freq_failure(hfreq,minf0,maxf0)
    hfreq,hmag,hphase = UM.mag_interpolate(hfreq,hfreqNoise, hmag, hphase)

    ## tst
    # save_path = 'result/test.xlsx'
    # UM.save_matrix(save_path, hfreq, sheetName='guitar_freq_nonsilence_interp')

    nF = hfreq.shape[0]
    sdInfo['nF'] = nF

    # calculate F0 (mean frequency of fundamentals)
    f0 = np.mean(hfreq[:,0])
    sdInfo['f0']=f0

    # find frequency related features
    freqMean, freqVar = UM.freq_ana(hfreq, f0, 0)
    freqVarRate = 0.01
    freqSmoothLen = 31
    sdInfo['freqInterval'] = 1
    sdInfo['freqMean'] = freqMean
    sdInfo['freqVar'] = freqVar
    sdInfo['freqVarRate'] = freqVarRate
    sdInfo['freqSmoothLen'] = freqSmoothLen

    # find magnitude related features
    t = np.arange(hfreq.shape[0]) * x.shape[0] / fs / hfreq.shape[0]
    tArray = np.array([t]).T
    hns, hCurve, hPoints = UM.find_curved_partial(hmag.T, tArray)
    for i in range(nH):
        # hP = np.concatenate((np.array([0]), hPoints[i, :], np.array([nF - 1])))
        if i == 0:
            magPointsValue = hmag[hPoints[i, :],i]
        else:
            magPointsValue = np.vstack((magPointsValue,hmag[hPoints[i, :],i]))

    sdInfo['magADSRIndex'] = hPoints.T
    sdInfo['magADSRValue'] = magPointsValue.T
    sdInfo['magADSRN']= hns.T

    # find phase related features
    ffphase = hphase[0, :]

    ## tst
    # plt.plot(ffphase,'x-')
    # plt.xlabel('Harmonic Number')
    # plt.ylabel('Phase(wrapped)')
    # plt.title('First Frame Phase of '+sdInfo['instrument']+sdInfo['pitch'])
    # plt.show()

    ffphaseSyn, ffpSlot, ffpIntercept, ffpErr = UM.find_first_frame_phase(ffphase, mode='linear')
    sdInfo['phaseffSlope'] = ffpSlot
    sdInfo['phaseffIntercept'] = ffpIntercept

    return sdInfo

def sound_syn_from_para(sdInfo):
    # synthesize the frequencies
    hfreqSyn = UM.freq_syn_from_para(sdInfo['nH'],sdInfo['nF'],sdInfo['f0'],sdInfo['freqInterval'], sdInfo['freqMean'],
                                     sdInfo['freqVar'], sdInfo['freqVarRate'], sdInfo['freqSmoothLen'])

    # print(hfreqSyn.shape)

    # synthesize the magnitude spectrum
    hmagSyn = UM.mag_syn_from_para(sdInfo['fs'],sdInfo['nH'],sdInfo['nF'],sdInfo['hopSize'],sdInfo['magADSRIndex'],sdInfo['magADSRValue'],
                                   sdInfo['magADSRN'])

    # save_path = 'result/test.xlsx'
    # UM.save_matrix(save_path, hfreqSyn, sheetName='guitar_freq_syn')


    # synthesize the phase spectrum
    hphaseSyn = UM.phase_syn_from_para(sdInfo['fs'],sdInfo['hopSize'],sdInfo['nH'],sdInfo['nF'],sdInfo['phaseffSlope'],
                                       sdInfo['phaseffIntercept'],hfreqSyn)
    # print(hphaseSyn.shape)

    # hmagSyn = hmagSyn-5

    y, yh, yst = HPS.hpsModelSynth(hfreqSyn, hmagSyn, hphaseSyn, sdInfo['stocEnv'], sdInfo['FFTLenSyn'], sdInfo['hopSize'], sdInfo['fs'])

    return y, yh, yst, hfreqSyn, hmagSyn, hphaseSyn

def save_features_to_csv():
    instruments = ['flute', 'oboe', 'saxophone', 'horn', 'violin', 'erhu', 'guitar', 'harp', 'piano', 'organ','trumpet']
    pitches = ['G3','C4','G4','A4','C5','G5','C6','G6']
    for instrument in instruments:
        for pitch in pitches:
            file_path = '../../sounds/sd/' + instrument + '-' + pitch + '.wav'
            if os.path.exists(file_path):
                source = ''
                index = 0

                # parameters for harmonic model　
                nH = 40
                minf0 = 400
                maxf0 = 500
                M = 4001
                N = 8192
                Ns = 512
                H = 128

                sdInfo = {
                    'instrument': instrument,
                    'pitch': pitch,
                    'source': source,
                    'index': index,
                    'nH': nH,
                    'nF': 0,
                    'FFTLenAna': N,
                    'FFTLenSyn': Ns,
                    'hopSize': H
                }  # general info of the sound

                sdInfo = sound_info_clct(file_path, sdInfo, nH=nH, minf0=minf0, maxf0=maxf0, M=M, N=N, Ns=Ns, H=H)

                save_path = 'result/features'
                UM.save_dictionary(sdInfo, save_path)


def save_features_to_json():
    instruments = ['flute', 'oboe', 'saxophone', 'horn', 'violin', 'erhu', 'guitar', 'harp', 'piano', 'organ','trumpet']
    pitch = 'A4'
    for instrument in instruments:
        file_path = '../../sounds/sd/' + instrument + '-A4' + '.wav'
        if os.path.exists(file_path):
            source = ''
            index = 0

            # parameters for harmonic model　
            nH = 40
            minf0 = 400
            maxf0 = 500
            M = 4001
            N = 8192
            Ns = 512
            H = 128

            sdInfo = {
                'instrument': instrument,
                'pitch': pitch,
                'source': source,
                'index': index,
                'nH': nH,
                'nF': 0,
                'FFTLenAna': N,
                'FFTLenSyn': Ns,
                'hopSize': H
            }  # general info of the sound

            sdInfo = sound_info_clct(file_path, sdInfo, nH=nH, minf0=minf0, maxf0=maxf0, M=M, N=N, Ns=Ns, H=H)

            save_path = 'result/features'
            UM.save_dictionary(sdInfo, save_path,1)

def naming_info_phiharmonia(str):
    # extracting instrument information from the names
    div = [pos for pos,char in enumerate(str) if char == '_']
    instrument = str[:div[0]]
    pitch = str[div[0]+1: div[1]]
    length = str[div[1]+1: div[2]]
    dynamic = str[div[2]+1: div[3]]
    articulation = str[div[3]+1: str.find(".")]
    return instrument, pitch, length, dynamic, articulation


if __name__ == "__main__":

    exp = 4 

    if exp == 0:
        sdInfo = UM.read_features('result/features/flute_acoustic/flute_acoustic-A5-002.json', 1)
        t = np.arange(sdInfo['stocEnv'].shape[0])
        ind = np.repeat(np.array([np.arange(sdInfo['stocEnv'].shape[1])]),sdInfo['stocEnv'].shape[0],axis=0)
        UM.plot_spec3d(ind,sdInfo['stocEnv'],t)

    elif exp == 1:
        ######## harmonics modification ################################################################################
        instruments = ['flute', 'oboe', 'saxophone', 'horn', 'violin', 'erhu', 'guitar', 'harp', 'piano', 'organ', 'trumpet']
        file_path = '../../sounds/A4/trumpet-A4.wav'

        'EH', 'OH', 'exp','gaussian','invGaussian',  'single'
        spec_modi(file_path,1,'exp',0)
        ################################################################################################################

    elif exp == 2:
        ######## sound analysis and synthesis ################################################################################
        # instruments = ['flute', 'oboe', 'saxophone', 'horn', 'violin', 'erhu', 'guitar', 'harp', 'piano', 'organ', 'trumpet']
        file_path = '../../sounds/phiharmonia/test/test_violin/violin_A3_1_forte_arco-normal.wav' 
        # UM.plot_spectrogram(file_path)
        # file_path = '../../sounds/A4/violin-A4.wav'
        # file_path = '../../sounds/midi_synth/flute-C5-1.wav'
        pitch = 'A3'
        UM.plot_spectrogram(file_path)

        # f = UM.pitch2freq(97)
        # print(f)

        # parameters for harmonic model　
        nH = 10
        minf0 = UM.pitch2freq(pitch)-50
        maxf0 =  minf0 + 100

        x, fs, hfreq, hmag, hphase, stocEnv = hps_ana(file_path,nH=nH,minf0=minf0, maxf0=maxf0,N=16384,M=16384,H = 256)
        # print(hfreq[:, 0], '\n', hmag[:, 0], '\n', hphase[:, 0])
        # print(hfreq[::,0],hfreq[::,1])
        # plt.plot(np.unwrap(hphase[0,:]),'x-')
        # plt.plot(x)

        # hphasetemp = np.concatenate((np.zeros([1, hphase.shape[1]]), hphase))
        # hphase = np.unwrap(hphasetemp, axis=0)[1:, :]

        # Showtest
        # 0: no result display
        # 1: sound
        # 2: graph
        # 3: sound+graph
        # 4: graphs of phase
        # Mode
        # 1: change frequency
        # 2: change magnitude
        # 3: change phase
        # 4: change frequency and magnitude
        # 5: change magnitude and phase
        # 6: change frequency and phase
        # 7: change frequency, magnitude and phase, thus fully synthesized sound
        # Select
        # 0: keep all harmonics
        # 1: select some harmonics
        # Trim
        # 0: no changes
        # 1: apply denoise process to the spectrum
        hfreqSyn, hmagSyn, hphaseSyn = spec_synth(x, fs, hfreq, hmag, hphase, stocEnv,showTest=2, mode=0, minf0=minf0, maxf0=maxf0, select=0,trim=0)
        # print(hfreqSyn[:,5:],hmagSyn[:,5:],hfreqSyn[:,5:])
        # print(hfreqSyn[0,:])
        # plt.figure(2)
        # plt.subplot(1,2,1)
        # plt.plot(hphase[0, :], 'x-')
        # plt.subplot(1,2,2)
        # plt.plot(hmagSyn[0, :], 'x-')
        # plt.show()

        # harm = hmagSyn[:, 1]
        # smLength = 11
        # harm = UM.smooth(harm, smLength)


        ################################################################################################################

    elif exp == 3:
        ######## features collecting ################################################################################
        # instruments = ['flute', 'oboe', 'saxophone', 'horn', 'violin', 'erhu', 'guitar', 'harp', 'piano', 'organ', 'trumpet','saw','sine','pulse']
        instrument = 'brass'
        pitch = 'A4'
        source = ''
        index = 1

        # file_path = '../../sounds/sd/'+ instrument + '-' + pitch + '.wav'
        # file_path = '../../sounds/brass_acoustic/brass_acoustic_valid/brass_acoustic_006-069-075.wav'
        # save_path = 'result/features/midi_synth/'
        file_path = '../../sounds/A4/flute-A4.wav' 
        
        # parameters for harmonic model　
        nH = 40
        minf0 = UM.pitch2freq('A4')-50
        maxf0 = UM.pitch2freq('A4')+50 
        M = 4001
        N = 8192
        Ns = 512
        H = 128

        sdInfo = {
            'instrument': instrument,
            'pitch': pitch,
            'source': source,
            'index': index,
            'nH': nH,
            'nF':0,
            'FFTLenAna':N,
            'FFTLenSyn':Ns,
            'hopSize':H
        }  # general info of the sound

        sdInfo = sound_info_clct(file_path, sdInfo, nH=nH, minf0=minf0, maxf0=maxf0, M=M, N=N, Ns=Ns, H=H)
        #
        #         # sdInfo['f0'] = sdInfo['f0']/2

        for i in sdInfo:
            print(i,':',sdInfo[i],'\n')

        # modify rise time
        # magADSRIndex = sdInfo['magADSRIndex']
        # riseTime = magADSRIndex[1,:]-magADSRIndex[0,:]
        # riseTimeModi = copy.deepcopy(riseTime)
        # riseTimeModi[3:] = np.round(riseTime[3:]+200)
        # magADSRIndex[0,3:] = magADSRIndex[0,3:]+50
        # magADSRIndex[1,:] = magADSRIndex[0,:]+riseTimeModi
        # sdInfo['magADSRIndex'] = magADSRIndex

        y, yh, yst, hfreqSyn, hmagSyn, hphaseSyn = sound_syn_from_para(sdInfo)


        # Showtest
        # 0: no display
        # 1: sound
        # 2: graph
        # 3: sound+graph
        UM.display_syn(file_path,sdInfo['fs'], sdInfo['hopSize'],sdInfo['nF'], y,yh,yst, hfreqSyn, hmagSyn, hphaseSyn, mode=1)
        plt.plot(hmagSyn[:,0])
        plt.show()


        # save features to a csv file
        # save_path = 'result/features'
        # UM.save_dictionary(sdInfo,save_path,1)
        # UM.save_dictionary(sdInfo,save_path,1)

        # sdInfo = sound_info_clct(file_path, sdInfo, nH=nH, minf0=minf0, maxf0=maxf0, M=M, N=N, Ns=Ns, H=H)
        # UM.save_dictionary(sdInfo, save_path, fm=1)
        # for k in sdInfo:
        #     print(k,':',sdInfo[k])
        # print('done')

        ################################### #############################################################################

    ################################ collect features ####################################
    elif exp == 4:
        instrument = 'reed_acoustic'
        files_path = '../../sounds/reed_acoustic/reed_acoustic_valid/'
        save_path = 'result/features/reed_acoustic/reed_acoustic_valid/'

        # official features file
        if os.path.exists(files_path+'examples.json'):
            with open(files_path+'examples.json','r') as f:
                example = json.load(f)
        else:
            print('example.json not exists!')

        # collect features of flutes
        files = os.listdir(files_path)
        for f in files:
            print(f)
            if f[-3:] == 'wav':
                if f[:-4] in example.keys():
                    ft = example[f[:-4]]

                    # change pitch number to pitch name
                    chroma = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
                    pitch = ft['pitch']

                    if pitch < 72:
                        M = 4096
                    else:
                        M = 2048

                    pitch = chroma[pitch%12] + str(int(pitch/12)-1)

                    N = M*2
                    Ns = 512
                    H = 256
                    nH = 40
                    sdInfo = {
                        'instrument':instrument,
                        'pitch':pitch,
                        'source':'NSynth_'+ft['note_str'] ,
                        'index':ft['instrument_str'][-3:]+'-'+ft['note_str'][-3:],
                        'nH':nH,
                        'nF': 0,
                        'FFTLenAna': N,
                        'FFTLenSyn': Ns,
                        'hopSize': H
                    }

                    freqency = UM.pitch2freq(sdInfo['pitch'])
                    minf0 = freqency-50
                    maxf0 = minf0+100

                    try:
                        sdInfo = sound_info_clct(files_path+f, sdInfo, nH=nH, minf0=minf0, maxf0=maxf0, M=M, N=N, Ns=Ns, H=H)
                        UM.save_dictionary(sdInfo, save_path, fm=1)
                        print(sdInfo['pitch'],ft['note_str'],' saved!')
                    except:
                        print('Failed to collect features for',sdInfo['pitch'])

                else:
                    print('not a recorded sound!')

    #################################################################################################

    ################################ synthesis ####################################
    elif exp == 5:
        # nH = 20
        # nF = 810
        # sdInfo = {
        #     'instrument' : 'syn',
        #     'pitch':'A4',
        #     'source':'',
        #     'index':0,
        #     'nH':nH,
        #     'nF':nF,
        #     'FFTLenAna': 8192,
        #     'FFTLenSyn': 512,
        #     'hopSize': 128,
        #     'fs':44100,
        #     'f0':440,
        #     'freqInterval':1,
        #     'freqMean':np.zeros(nH),
        #     'freqVar':np.zeros(nH),
        #     'freqVarRate':0,
        #     'freqSmoothLen':31
        # }
        #
        # # magnitude spectrum
        # magADSRIndex = np.repeat(np.array([[10,100,700,800]]).T,nH,axis=1)
        #
        # magADSRValue = np.zeros((4,nH))
        # magADSRValue[1,:]=np.linspace(-40,-150,20)
        # magADSRValue[0, :] = magADSRValue[1,:] - 20
        # magADSRValue[2, :] = magADSRValue[1, :] - 3.1
        # magADSRValue[3, :] = magADSRValue[1, :] - 20
        #
        # magADSRN = np.repeat(np.array([[1,3,1.2,0.5,1]]).T, nH, axis=1)
        #
        # sdInfo['magADSRIndex'] = magADSRIndex
        # sdInfo['magADSRValue'] = magADSRValue
        # sdInfo['magADSRN'] = magADSRN
        #
        # # phase spectrum
        # sdInfo['phaseffSlope'] = -20
        # sdInfo['phaseffIntercept'] = 45
        #
        # # stochastic model
        # sdInfo['stocEnv'] = -60*np.ones((nF,12))

        flInfo = UM.read_features('result/features/phiharmonia/violin/violin-81-4.json', 1)
        # sdInfo = flInfo.copy()
        # sdInfoNew = sdInfo.copy()
        sdInfo = copy.deepcopy(flInfo)
        sdInfoNew = copy.deepcopy(sdInfo)
        # sdInfo['magADSRValue'] = flInfo['magADSRValue'][:,0:nH]
        #
        # sdInfo['magADSRIndex'] = (sdInfo['nF']*flInfo['magADSRIndex'][:, 0:nH]/flInfo['nF']).astype(int)
        #
        # sdInfo['magADSRN'] = flInfo['magADSRN'][:,0:nH]
        #
        # sdInfo['stocEnv'] = flInfo['stocEnv'][:, 0:nH]

        # Some Modifications
        # 1 change rise time separatly
        # magIndex = sdInfo['magADSRIndex']
        # nH = sdInfo['nH']
        # EOA = np.round(magIndex[1,:]-magIndex[0,:])*(1.5**(-np.arange(nH)+20)) + magIndex[0,:]
        # sdInfoNew['magADSRIndex'][1,:] = EOA

        # synthesis
        y, yh, yst, hfreqSyn, hmagSyn, hphaseSyn = sound_syn_from_para(sdInfo)
        yNew, yhNew, ystNew, hfreqSynNew, hmagSynNew, hphaseSynNew = sound_syn_from_para(sdInfoNew)

        # playsound
        mode = 1
        if mode in [1,3]:
            outputFileSines = 'output_sounds/syn_sines.wav'
            outputFileStochastic = 'output_sounds/syn_stochastic.wav'
            outputFile = 'output_sounds/syn.wav'
            outputFileNew = 'output_sounds/synNew.wav'
            UF.wavwrite(yh, sdInfo['fs'], outputFileSines)
            UF.wavwrite(yst, sdInfo['fs'], outputFileStochastic)
            UF.wavwrite(y, sdInfo['fs'], outputFile)
            UF.wavwrite(yNew,sdInfo['fs'], outputFileNew)

            UF.wavplay('../../sounds/phiharmonia/violin/violin_A5_15_forte_arco-normal.wav')
            UF.wavplay(outputFile)
            UF.wavplay(outputFileNew)


        if mode in [2,3]:
            t = np.arange(sdInfo['nF']) * sdInfo['hopSize'] / sdInfo['fs']
            plt.figure()
            UM.plot_spec3d(hfreqSyn, hmagSyn, t, 1)
            plt.figure()
            UM.plot_spec3d(hfreqSynNew, hmagSynNew, t, 1)
            # plt.figure()
            # UM.plot_spec3d(hfreqSyn, hphaseSyn, t, 1)

        ###############################################################################
    
    elif exp == 6:
        ##################### collect features for sounds in phiharmonia db ##################
        instruments = ['oboe','clarinet','saxophone','french horn','trumpet','violin','cello']
        # instruments = ['saxophone','french horn','trumpet','violin','cello']
        # instruments = ['test']
        files_path = '../../sounds/phiharmonia/'
        save_path = 'result/features/phiharmonia/'
        for ins in instruments:
            for f in os.listdir(files_path+ins+"/"):
                if f[-3:] == 'wav':
                    # get genral information for the sound
                    instrument, pitch, length, dynamic, articulation = naming_info_phiharmonia(f)
                    length_all = ['15','1','05','025']
                    dynamic_all = ['pianissimo','piano','mezzo-piano','mezzo-forte','forte','fortissimo']
                    
                    if f[-3:] == 'wav' and dynamic in dynamic_all:
                        index = length_all.index(length)*len(dynamic_all) + dynamic_all.index(dynamic) 

                        # change pitch name to number
                        chroma = ['C','Cs','D','Ds','E','F','Fs','G','Gs','A','As','B']
                        pitch = chroma.index(pitch[:-1]) + (int(pitch[-1])+1)*12

                        ## tst
                        # print(pitch)

                        if pitch < 72:
                            M = 4096
                        else:
                            M = 2048


                        N = M*2
                        Ns = 512
                        H = 256
                        nH = 40
                        sdInfo = {
                            'instrument':instrument,
                            'pitch':str(pitch),
                            'source':'phiharmonia_database' ,
                            'index':index,
                            'nH':nH,
                            'nF': 0,
                            'FFTLenAna': N,
                            'FFTLenSyn': Ns,
                            'hopSize': H
                        }

                        freqency = UM.pitch2freq(pitch)
                        minf0 = freqency-50
                        maxf0 = freqency+50

                        try:
                            sdInfo = sound_info_clct(files_path+ins+'/'+f, sdInfo, nH=nH, minf0=minf0, maxf0=maxf0, M=M, N=N, Ns=Ns, H=H)
                            UM.save_dictionary(sdInfo, save_path+ins+'/', fm=1)
                            print(f,'saved!')
                        except:
                            print('Failed to collect features for',f)

                    else:
                        print(f,'is not a recorded sound or has wrong dynamic!')
                    
            
    #################################################################################################
    

