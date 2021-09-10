import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../transformations/'))
sys.path.append('/Library/Python/2.7/site-packages/')

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.interpolate as itpl
import scipy.stats as stats
from scipy.optimize import curve_fit, least_squares
import math

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

def model_beginning_segment(nonSilence, harm):
    # beginning segment: start to 50% of max
    # harmMax = np.max(harm)
    # harmMaxIndex = np.argmax(harm)
    # beginningEnd = nonSilence[0]+1
    # print(harmMax,harmMaxIndex)
    # for i in range(nonSilence[0]+1,harmMaxIndex+1):
    #     if harm[i] >= harmMax + 20 * math.log10(0.5): # threshold: 50% of max
    #         beginningEnd = i
    #         print('i:',i)
    #         break
    # beginningSeg = [nonSilence[0],beginningEnd]
    # beginningSeg = [nonSilence[0],100]
   
    # beginning segment: 
    harm_sm = find_spline_harmonic(harm,lam = 1e3)
    harm_sm_deri = np.diff(harm_sm)
    # beginningEnd = min(np.argwhere(np.diff(np.sign(harm_sm_deri)) < 0).flatten()[0]+1,100)   # maximum possible end index is 100
    beginningEnd = min(np.argwhere(np.abs(harm_sm_deri) < 0.15).flatten()[0]+1,100) # derivative below s threshold: 0.1
    beginningSeg = [nonSilence[0],beginningEnd]

    # polynomial regression to beginning segment
    beginningPara = np.polyfit(np.arange(beginningSeg[1]-beginningSeg[0]+1),harm[beginningSeg[0]:beginningSeg[1]+1],1) 

    return beginningSeg, beginningPara

def exp_fitting_func(x,a,b,c):
    return a*np.exp(-b*x)+c

def model_steady_segment(nonSilence, beginningSeg, harm):
    # steady segment: mid-third of the sound
    steadySeg = [beginningSeg[1]+1,round((nonSilence[0]+3*nonSilence[1])/4)]

    # 1 exponential regression to steady segment : y = A*exp(x) + B
    # x = np.arange(steadySeg[1]-steadySeg[0]+1)
    # y = harm[steadySeg[0]:steadySeg[1]+1]
    # p0 = [3,2,-40]
    # steadyPara, exp_cov = curve_fit(exp_fitting_func, x, y, p0=p0, maxfev = 100000000) 

    # 2 polynomial regression
    steadyPara = np.polyfit(np.arange(steadySeg[1]-steadySeg[0]+1),harm[steadySeg[0]:steadySeg[1]+1],3)

    return steadySeg, steadyPara

def calculate_poly(x,para):
    order = para.size
    for i in range(order):
        if i == 0:
            res = para[i]*(x**(order-1))
        else:
            res = res + para[i]*(x**(order-i-1))
    return res

def model_EOA(harm, beginningSeg, beginningPara, steadySeg, steadyPara, displayErr = 0):
    # built the two curves
    attackRange = np.arange(beginningSeg[0],steadySeg[1]+1)
    beginningCv = calculate_poly(attackRange - beginningSeg[0], beginningPara)  
    steadyCv = calculate_poly(attackRange - steadySeg[0], steadyPara)
    
    # detect intersection by checking equal values
    intersect = np.argwhere(np.diff(np.sign(beginningCv-steadyCv))).flatten()
    
    ## tst
    # print("intersections: ", intersect)

    EOA_ind = 0
    if intersect.size == 0:
        EOA_ind = beginningSeg[1]+1
        if displayErr == 1:
            plt.plot(harm)
            plt.plot(attackRange, beginningCv, 'k-')
            plt.plot(attackRange, steadyCv, 'b-')
            plt.plot(attackRange[EOA_ind],beginningCv[EOA_ind],'ro')
            plt.title('harm #'+str(h))
            plt.show()

    elif intersect.size >1:
        EOA_ind = intersect[1]+1
    else:
        EOA_ind = intersect[0]+1

    return EOA_ind

if __name__ == "__main__":

    exp = 5 

    if exp == 0:
        # file_path = '../../sounds/midi_synth/piano-C4-1.wav'
        # file_path = '../../sounds/A4/trumpet-A4.wav' 
        file_path = '../../sounds/brass_acoustic/brass_acoustic_valid/brass_acoustic_006-081-075.wav'
        # HpS model 
        nH = 40
        minf0 = UM.pitch2freq('C4') - 50
        maxf0 = minf0 + 100
        N = 4096
        M = 4096
        H = 256
        x, fs, hfreq, hmag, hphase, stocEnv = TM.hps_ana(file_path, nH=nH, minf0=minf0, maxf0=maxf0, N=N, M=M, H=H)
       
        # find non-silent region
        nonSilence = UM.non_silence_dtct(hfreq[:,0],minf0,maxf0) 
        # x = np.arange(30) 
        # y = 3*np.exp(-0.2*x)+3 + 0.2*np.random.rand(30)
        
        steadySeg = [round((nonSilence[1]+2*nonSilence[0])/3),round((nonSilence[0]+2*nonSilence[1])/3)]
        x = np.arange(steadySeg[1]+1-steadySeg[0])
        y = hmag[steadySeg[0]:steadySeg[1]+1,0]
        p0 = [5,3,-40]
        para, cov = curve_fit(exp_fitting_func,x,y,p0 = p0)
        plt.plot(x,y,'k-')
        plt.plot(x, exp_fitting_func(x,para[0],para[1],para[2]),'b-')
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
            t,y,y_spl,dy_spl= find_attack_univariateSpline(harm,lam = 1e4)

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
        # file_path = '../../sounds/midi_synth/piano-C4-1.wav'

        nH = 20
        minf0 = UM.pitch2freq(69) - 50
        maxf0 = UM.pitch2freq(69) + 50

        x, fs, hfreq, hmag, hphase, stocEnv = TM.hps_ana(file_path, nH=nH, minf0=minf0, maxf0=maxf0, N=4096, M=4096)

        hfreqSyn = hfreq.copy()
        hmagSyn = hmag.copy()
        hphaseSyn = hphase.copy()

        for i in range(nH):
            harm = hmag[:,i]

            # linear smoothing
            # smLength = 11
            # harm = UM.smooth(harm, smLength)
            harmZero = -200 * (harm == 0)
            harm = harm + harmZero
            # scipy.interpolation.univariatespline
            hmag[:,i] = harm
            hmagSyn[:,i] = find_spline_harmonic(harm,lam = 1e5)


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
        # UF.wavplay(file_path)
        # UF.wavplay(outputFile)

        # display harmonics
        # t = np.arange(hfreq.shape[0]) * x.shape[0] / fs / hfreq.shape[0]
        # plt.figure()
        # UM.plot_spec3d(hfreq, hmag, t, 1)
        # plt.figure()
        # UM.plot_spec3d(hfreqSyn, hmagSyn, t, 1)

        # plotting harmonics
        t = np.arange(hmag.shape[0])*H/float(fs)
        plt.plot(t,hmag[:,1],t,hmagSyn[:,1])
        plt.show()


    elif exp == 3: # transient detection test
        # instruments = ['flute', 'oboe', 'saxophone', 'horn', 'violin', 'erhu', 'guitar', 'harp', 'piano', 'organ', 'trumpet']
        file_path = '../../sounds/A4/trumpet-A4.wav'
        # file_path = '../../sounds/mridangam.wav'

        # plot spectrogram
        plt.figure()
        mX,pX,fs,H = UM.plot_spectrogram(file_path)
        # print(mX.shape) # (142,2049) when N = 4096
        print(np.max(pX))

        # find transient level
        transLevel = find_transient_level(UM.dB2abslt(mX))
        # print(transLevel[0:40])

        # plotting
        t = np.arange(transLevel.size)*H/fs
        
        plt.plot(t,transLevel)
        plt.show()         
    
    # calculate some energy
    elif exp == 4:
        # load the sound
        # instruments = ['flute', 'oboe', 'saxophone', 'horn', 'violin', 'erhu', 'guitar', 'harp', 'piano', 'organ', 'trumpet']
        file_path = '../../sounds/midi_synth/piano-C4-1.wav'
        # file_path = '../../sounds/A4/trumpet-A4.wav'
        nH = 40
        minf0 = UM.pitch2freq('C4') - 50
        # maxf0 = UM.pitch2freq('C5') + 50
        maxf0 = minf0+100
        N = 4096
        M = 4096
        H = 256

        x, fs, hfreq, hmag, hphase, stocEnv = TM.hps_ana(file_path, nH=nH, minf0=minf0, maxf0=maxf0, N=N, M=M, H=H)

        # plot spectrogram
        mX,pX,fs,H = UM.plot_spectrogram(file_path)
        
        # dbg
        print('spectrogram shape:', mX.shape)
        print('hmag shape:',hmag.shape)
        
        # calculate multiple energy for every frame: E, E_harm, E_noise_all, E_noise_high, E_noise_low, HFC
        frameNum, binNum = mX.shape
        E = np.empty(0)
        E_harm = np.empty(0)
        E_noise_all = np.empty(0)
        E_noise_high = np.empty(0)
        E_noise_low = np.empty(0)
        HFC = np.empty(0)

        # decide the range of high frequency content
        highestHarmFreq = np.median(hfreq[:,-1])  # median of the highest harmonic, aka the approximate upperbound of harmonics
        highNoiseLB = int(highestHarmFreq * N / fs)
        mX_high = mX[:,highNoiseLB:]
        mX_low = mX[:,:highNoiseLB]
        
        # dbg
        print('highest harmonics frequency:', highestHarmFreq, 'high noise lower bound:', highNoiseLB)
            
        for f in range(frameNum):
            # E
            efr = frame_energy(UM.dB2abslt(mX[f,:]))
            E = np.append(E, efr)
            
            # E harmonics
            harm = hmag[f,:]
            harmZero = -200 * (harm == 0)
            harm = harm + harmZero
            eHarmfr = frame_energy(UM.dB2abslt(harm))            
            E_harm = np.append(E_harm, eHarmfr)

            # E all noise
            E_noise_all = np.append(E_noise_all, efr - eHarmfr)

            # E high frequency content (noise)
            eNoiseHigh = frame_energy(UM.dB2abslt(mX_high[f,:]))
            E_noise_high = np.append(E_noise_high, eNoiseHigh)

            # E low frequency noise
            eNoiseLow = frame_energy(UM.dB2abslt(mX_low[f,:])) - eHarmfr
            E_noise_low = np.append(E_noise_low,eNoiseLow)

            # HFC
            hfcfr = frame_high_frequency_content(UM.dB2abslt(mX[f,:]))
            HFC = np.append(HFC, hfcfr)
        
        # plotting
        t = np.arange(frameNum)*H/float(fs)
        plt.plot(t,E,t,E_harm,t,E_noise_all,t,E_noise_high,t,E_noise_low)
        plt.legend(('E','E_harm','E_noise_all','E_noise_high','E_noise_low'))
        plt.title('harmonics energy')
        plt.show()
        
        # dbg
        plt.plot(t,E_noise_high)
        plt.title('E_noise_high')
        plt.show()

        plt.plot(t,HFC),
        plt.title('HFC')
        plt.show()
        # dbg end


    elif exp == 5: # detecting the end of attack by checking the intersection of modeling begining segment and steady segment
        # load the sound
        # file_path = '../../sounds/midi_synth/piano-C4-1.wav'
        # file_path = '../../sounds/midi_synth/flute-C5-1.wav'
        # file_path = '../../sounds/A4/trumpet-A4.wav'
        # file_path = '../../sounds/A4/flute-A4.wav'
        # file_path = '../../sounds/A4/oboe-A4.wav'
        # file_path = '../../sounds/A4/violin-A4.wav'
        # file_path = '../../sounds/A4/horn-A4.wav'
        file_path = '../../sounds/brass_acoustic/brass_acoustic_valid/brass_acoustic_006-081-075.wav'
        pitch = 81
        # HpS model 
        nH = 10
        minf0 = UM.pitch2freq(pitch) - 50
        maxf0 = minf0 + 100
        N = 4096
        M = 4096
        H = 256
        x, fs, hfreq, hmag, hphase, stocEnv = TM.hps_ana(file_path, nH=nH, minf0=minf0, maxf0=maxf0, N=N, M=M, H=H)
       
        # find non-silent region
        nonSilence = UM.non_silence_dtct(hfreq[:,0],minf0,maxf0) 

        # end of attack detection
        EOA = []
        NOIntersect = []
        begSegs = []
        for h in range(nH):
            harm = hmag[:,h]
            harmZero = -200 * (harm == 0)
            harm = harm + harmZero
            harm_sm = find_spline_harmonic(harm,lam = 1e3)
            
            # beginning segment
            beginningSeg, beginningPara = model_beginning_segment(nonSilence, harm)
            # tst
            print('harm #',h,':')
            print('beginning segment:', beginningSeg, ' parameters:',beginningPara)
            
            # steady segment
            # try:
            steadySeg, steadyPara = model_steady_segment(nonSilence,beginningSeg, harm)
            print('steady segment:', steadySeg, ' parameters:',steadyPara)
            # except:
            #     plt.plot(harm)
            #     plt.show()
            #     print('exponential fitting failed...do linear regression instead')
            #     steadySeg = [beginningSeg[1]+10,round((nonSilence[0]+3*nonSilence[1])/4)]
            #     std_slope, std_intercept, _,__,___ =stats.linregress(np.arange(steadySeg[0],steadySeg[1]+1),harm[steadySeg[0]:steadySeg[1]+1])
            #     std_para = [std_slope,std_intercept]
            
            ## find the intersection of two curves
            # built the two curves
            attackRange = np.arange(beginningSeg[0],steadySeg[1]+1)
            beginningCv = calculate_poly(attackRange - beginningSeg[0], beginningPara)  
            steadyCv = calculate_poly(attackRange - steadySeg[0], steadyPara)
            
            # detect intersection by checking equal values
            intersect = np.argwhere(np.diff(np.sign(beginningCv-steadyCv))).flatten()
            
            ## tst
            print('intersection: ', intersect)

            EOA_ind = 0
            if intersect.size == 0:
                plt.plot(harm)
                plt.plot(attackRange, beginningCv, 'k-')
                plt.plot(attackRange, steadyCv, 'b-')
                plt.plot(attackRange[EOA_ind],beginningCv[EOA_ind],'ro')
                plt.title('harm #'+str(h))
                plt.show()
                EOA_ind = beginningSeg[1]+1
                NOIntersect.append(h)

            elif intersect.size >1:
                EOA_ind = intersect[1]+1
                print('EOA index:',attackRange[EOA_ind], intersect.size)
            else:
                EOA_ind = intersect[0]+1
                print('EOA index:',attackRange[EOA_ind])

            EOA.append(EOA_ind)
            begSegs.append(beginningSeg)

            # plot the fitting result
            if h < 10:
                plt.subplot(1,2,1)
                plt.plot(harm)
                plt.plot(attackRange[:EOA_ind+1], beginningCv[:EOA_ind+1], 'k-')
                plt.plot(attackRange[EOA_ind:], steadyCv[EOA_ind:], 'b-')
                plt.plot(attackRange[EOA_ind],beginningCv[EOA_ind],'ro')
                plt.plot(beginningSeg[1],harm[beginningSeg[1]],'go')
                plt.title('harm #'+str(h))

                plt.subplot(1,2,2)
                harm_diff = np.diff(harm)
                plt.plot(harm_diff)
                plt.plot(beginningSeg[1],harm_diff[beginningSeg[1]], 'go')
                plt.title('harm # '+str(h)+' derivative')
                plt.show()


            #if beginningSeg[1] > 50:
            #    print(harm_diff)
                
        print('EOA:',EOA)
        print('no intersection harms:', NOIntersect)
        print('Beginning Segments:',begSegs)

    
    elif exp == 6:
        # test UM.find_EOA function
        file_path = '../../sounds/brass_acoustic/brass_acoustic_valid/brass_acoustic_006-081-075.wav'
        # file_path = '../../sounds/A4/horn-A4.wav'
        pitch = 81
        # HpS model 
        nH = 10
        minf0 = UM.pitch2freq(pitch) - 50
        maxf0 = minf0 + 100
        N = 4096
        M = 4096
        H = 256
        x, fs, hfreq, hmag, hphase, stocEnv = TM.hps_ana(file_path, nH=nH, minf0=minf0, maxf0=maxf0, N=N, M=M, H=H)
        nonSilence = UM.non_silence_dtct(hfreq[:,0],minf0,maxf0) 

        # end of attack detection
        EOA = []
        NOIntersect = []
        begSegs = []
        
        for h in range(nH):
            harmEOA, beginningSeg, steadySeg, harm_sm = UM.find_EOA(hmag[:,h])
            EOA.append(harmEOA)
            begSegs.append(beginningSeg)
        
        print('EOA:',EOA)
        print('Beginning Segments:',begSegs)

