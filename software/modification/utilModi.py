# utility functions for modifications

import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))

import numpy as np
import pandas as pd
import math
import csv
import json

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm, linregress
import scipy.optimize as opt
import scipy.signal as sig

import utilFunctions as UF
import stft as STFT

########## functions for spectral descriptor tests  ###############
def evenAtten(hmag, attenRatio):
    hmagAttened = hmag.copy()
    for h in range(hmag.shape[1]):
        if h%2 == 1:
            hmagAttened[::,h]= hmagAttened[::,h] + 20*math.log10(attenRatio)

    return hmagAttened

def oddAtten(hmag, attenRatio, keepF0):
    hmagAttened = hmag.copy()
    for h in range(hmag.shape[1]):
        if h % 2 == 0:
            hmagAttened[::, h] = hmagAttened[::, h] + 20 * math.log10(attenRatio)

    if keepF0 == 1:
        hmagAttened[::,0] = hmag[::,0]

    return hmagAttened

def expAtten(hmag,rate):
    hmagAttened = hmag.copy()
    for h in range(hmag.shape[1]):
        hmagAttened[::, h] = hmagAttened[::, h]-20*rate*h*math.log10(math.e)

    return hmagAttened

def gaussianModi(hmag, hfreq, freqMean, freqVar,keepF0):

    th = -500

    hmagAttened = hmag.copy()
    atten = norm.pdf(hfreq, freqMean, freqVar)

    for fl in range(hmag.shape[0]):
        attenNorm = atten[fl,::] / np.max(atten[fl,::])
        hmagAttened[fl,::] = hmagAttened[fl,::] + 20*np.log10(attenNorm)
        for i in range(hmag.shape[1]):
            if hmagAttened[fl,i] < th:
                hmagAttened[fl, i] = th

    if keepF0 == 1:
        hmagAttened[::,0] = hmag[::,0]

    return hmagAttened

def invGaussianModi(hmag, hfreq, freqMean, freqVar,keepF0):

    th = -500
    esp = 0.000000001

    hmagAttened = hmag.copy()
    atten = norm.pdf(hfreq, freqMean, freqVar)
    print(atten)

    for fl in range(hmag.shape[0]):
        attenNorm = atten[fl,::] / np.max(atten[fl,::])
        w = 1 - attenNorm + esp
        hmagAttened[fl,::] = hmagAttened[fl,::] + 20*np.log10(w)
        for i in range(hmag.shape[1]):
            if hmagAttened[fl,i] < th:
                hmagAttened[fl, i] = th

    if keepF0 == 1:
        hmagAttened[::,0] = hmag[::,0]

    return hmagAttened

def singleModi(hmag, harmNo, rate):
    hmagAttened = hmag.copy()

    if harmNo <= hmag.shape[1]:
        hmagAttened[::,harmNo-1] = hmagAttened[::,harmNo-1] * rate

    return hmagAttened

#############################################################

########## functions for frequency synthesis  ###############

def non_zero_var(x):
    var = []
    for i in range(x.shape[1]):
         xF = x[::,i]
         xN = xF[np.nonzero(xF)]
         var.append(np.var(xN))

    return np.array(var)

def non_zero_mean(x,f0):
    mean = []
    for i in range(x.shape[1]):
         xF = x[::,i]
         xN = xF[np.nonzero(xF)]-f0*(i+1)
         mean.append(np.mean(xN))

    return np.array(mean)

def freq_ana(hfreq,f0,noiseRedu = 1):
    freqSlct = hfreq.copy()
    if noiseRedu == 1:
        # examing validation by difference with the previous harmony
        # freqPre = np.concatenate((np.zeros((hfreq.shape[0],1)),hfreq[::,::-2]),axis=1)
        # freqDrv = hfreq-freqPre
        # freqVld = (freqDrv>f0*0.8) & (freqDrv>f0*1.2) # threshold for validation of a certain frequency
        # freqSlct = hfreq * freqVld

        # examing validation by difference with standard frequency
        freqStd = np.repeat(np.array([np.arange(1,hfreq.shape[1]+1)]),hfreq.shape[0],axis=0) * f0
        freqDrv = hfreq-freqStd
        freqVld = (np.abs(freqDrv)<f0*0.2) # threshold for validation of a certain frequency
        freqSlct = hfreq * freqVld

    freqMean = non_zero_mean(freqSlct,f0)
    freqVar = non_zero_var(freqSlct)

    return freqMean, freqVar
#############################################################

########## functions for magnitude synthesis  ###############
def smooth(y, box_pts=21):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    leftPad = smooth_padding(y,box_pts)
    rightPad = smooth_padding(np.flip(y),box_pts)
    res = np.concatenate((leftPad,y_smooth,rightPad))
    return res

def smooth_padding(x, box):
    l = int((box-1)/2)
    y = np.zeros(l)
    for i in range(l):
        y[i] = np.sum(x[:int(i+(box+1)/2)])/(i+(box+1)/2)
    return y


# find adsr envelop
def find_adsr_perc(x):
    # find four time-value pairs: soa, eoa, sor, eor
    x_max = np.max(x)

    # #tst
    # print('x_max:',x_max)

    soa_ind = 0
    eoa_ind = 0
    sor_ind = x.shape[0]-1
    eor_ind = x.shape[0]-1
    # print(x_max,'\n',x)
    # soa: 10% of max from start
    # eoa: 90% of max from start
    fg = 0
    for i in range(x.shape[0]):
        if x[i] >= x_max + 20 * math.log10(0.1) and fg == 0:
            soa_ind = i
            fg = 1
        if x[i] >= x_max + 20*math.log10(0.6):
            eoa_ind = i
            break

    # sor: 10% of max from end
    # eor: 80% of max from end
    fg = 0
    for i in range(x.shape[0] - 1, 0, -1):
        if x[i] >= x_max + 20 * math.log10(0.1) and fg == 0:
            eor_ind = i
            fg = 1
        if x[i] >= x_max + 20*math.log10(0.6):
            sor_ind = i
            break

    # print(soa_ind, eoa_ind)
    # print(sor_ind, eor_ind)
    return [soa_ind, eoa_ind, sor_ind, eor_ind]

# find curve model for partials
def curve(n, values, x):
    if x.shape[0] == 0:
        return np.array([[0]])
    else:
        v0 = values[0][0]
        v1 = values[-1][0]
        return v0 + (v1 - v0) * np.power((1 - np.power((1 - (x - x.min()) / (x.max() - x.min())), n)), 1. / n)


def curve_err(n, values, x):
    if x.shape[0] == 0:
        return 0
    else:
        res = (values - curve(n, values, x))[:, 0]

        ##tst
        # print(res)

        return res


def find_opt_n(values, x, n0=1):
    if x.shape[0] == 0:
        return 1
    else:
        res = opt.least_squares(curve_err, n0, bounds=[-40, 40], args=(values, x))

        return res['x']


# find curve model for partials
def find_curved_partial(partial, ts):

    partial_n = []
    partial_curve = np.zeros(partial.shape)
    points_curve = np.zeros((partial.shape[0],4))
    pt_n = []

    for i in range(partial.shape[0]):
        pt = partial[i, :]

        # get rid of 0dB silence (set to -500dB)
        ptZero = -500 * (pt == 0)
        pt = pt + ptZero

        smLength = 31  # odd length
        pt_sm = smooth(pt, smLength)
        points = find_adsr_perc(pt_sm)
        points_curve[i,:] = np.array(points)
        points.insert(0, 0)
        points.append(pt_sm.shape[0]-1)

        ## tst
        if i in [0,1]:
            print(points)
            plt.plot(pt_sm)
            # plt.plot(pt)
            # plt.plot(ptZero)
            plt.plot(points,pt_sm[points],'x')
            plt.show()


        for j in range(5):
            st_ind = points[j]
            ed_ind = points[j + 1]
            if st_ind >= ed_ind-1:
                pt_n.append(1)
                if st_ind == ed_ind-1:
                    partial_curve[i,st_ind] = pt_sm[st_ind]
            else:
                n = find_opt_n(pt_sm[st_ind:ed_ind, np.newaxis], ts[st_ind:ed_ind])
                pt_n.append(n[0])
                simCurv = curve(n[0], pt_sm[st_ind:ed_ind, np.newaxis], ts[st_ind:ed_ind])
                partial_curve[i:i+1, st_ind:ed_ind] = simCurv.T

        partial_n.append(pt_n)
        pt_n = []

    partial_curve[::,-1] = partial_curve[::,-2]
    partial_n = np.array(partial_n)
    points_curve = points_curve.astype(int)

    return partial_n, partial_curve, points_curve

#############################################################

########## functions for plotting  ###############
def plot_spec3d(hfreq,hmag,t,content=0):
    ax_partial_curve = plt.axes(projection="3d")
    # fig = plt.figure()

    for i in range(hfreq.shape[1]):
        ax_partial_curve.plot(t, hfreq[::,i], hmag[::,i])

    if content == 1:
        ax_partial_curve.set_xlabel('time(sec)')
        ax_partial_curve.set_ylabel('frequency(Hz)')
        ax_partial_curve.set_zlabel('magnitude(dB)')
    elif content == 2:
        ax_partial_curve.set_xlabel('time(sec)')
        ax_partial_curve.set_ylabel('frequency(Hz)')
        ax_partial_curve.set_zlabel('phase(rad)')

    # ax_partial_curve.set_xlim((0,0.1))

    plt.show()

def plot_spec3d_cmp(hfreq1, hmag1, t1, hfreq2, hmag2, t2):

    plt.subplot(2,1,1)
    ax_partial_curve1 = plt.axes(projection="3d")
    # fig = plt.figure()
    for i in range(hfreq1.shape[1]):
        ax_partial_curve1.plot(t1, hfreq1[::, i], hmag1[::, i])

    plt.subplot(2, 1, 2)
    ax_partial_curve2 = plt.axes(projection="3d")
    for i in range(hfreq2.shape[1]):
        ax_partial_curve2.plot(t2, hfreq2[::, i], hmag2[::, i])

    plt.show()

def create_label(n):
    label = []
    for i in range(n):
        label.append(str(i+1))

    return label

#############################################################

########## excel ############################################
def save_matrix(save_path, matrix, sheetName=None, col=None, index=None):
    df = pd.DataFrame(matrix)
    if sheetName == None:
        sheet_name = 'unknown'
    with pd.ExcelWriter(save_path,mode='a') as writer:
        df.to_excel(writer, sheet_name=sheetName)
    return 0
###############################################################

########## functions for spectrum modification  ###############
def add_harm(hfreq, hmag, hphase, freqs,values,phases):
    f = np.mean(freqs)
    for i in range(hfreq.shape[1]):
        if f <= hfreq[0,i]:
            ind = i
            break
    if f > hfreq[0,-1]:
        ind = hfreq.shape[1]

    hfreqAdded = np.concatenate((hfreq[::,0:ind],freqs,hfreq[::,ind:]),axis=1)

    hmagAdded = np.concatenate((hmag[::,0:ind],values,hmag[::,ind:]),axis=1)

    hphaseAdded = np.concatenate((hphase[::, 0:ind], phases, hphase[::, ind:]), axis=1)

    return hfreqAdded, hmagAdded, hphaseAdded

def find_freq_spike(hfreq):
    fn, hn = hfreq.shape
    f0 = np.mean(hfreq[:, 0])
    # hfreqDeri = np.abs(hfreq - np.concatenate((hfreq[1:,:],hfreq[-1:,:]))) + np.abs(hfreq - np.concatenate((hfreq[0:1,:],hfreq[:-1,:])))
    # hfreqNoise = hfreqDeri > (f0 * thresholdRatio)

    # Threshold Controlling
    # Constant:
    thresholdRatio = 0.05*np.ones(hn)
    # Exponentially Increase:
    # thresholdRatio = 0.05 + np.exp(np.arange(hn)*0.02)-1

    hfreqNoise = []
    for i in range(hn):
        freqMin = np.mean(hfreq[:,i])-f0*thresholdRatio[i]
        freqMax = np.mean(hfreq[:,i])+f0*thresholdRatio[i]
        freqNoise = np.logical_or(hfreq[:,i]<=freqMin, hfreq[:,i]>=freqMax)
        hfreqNoise.append(freqNoise)

    hfreqNoise = np.array(hfreqNoise).T
    return hfreqNoise

def find_freq_failure(hfreq,minf0,maxf0):
    nH = hfreq.shape[1]
    freqNoise = np.logical_or(hfreq[:,0]<minf0, hfreq[:,0]>maxf0)
    # print(freqNoise)
    hfreqNoise = np.repeat(np.array([freqNoise]),nH,axis=0).T
    return hfreqNoise

def mag_interpolate(hfreq,hfreqNoise, hmag, hphase):
    fn, hn = hmag.shape
    hmagIntp = hmag.copy()
    hfreqIntp = hfreq.copy()
    hphaseIntp = hphase.copy()
    for i in range(hn):
        ptNoise = hfreqNoise[:,i]
        pt = hmag[:,i]
        ptf = hfreq[:,i]
        ptp = hphase[:, i]

        seg = []
        for j in range(fn):
            if ptNoise[j] == 1:
                seg.append(j)
                if j == fn-1:
                    res = intp(seg, pt[seg[0]-1], pt[seg[0]-1])
                    hmagIntp[seg[0]:seg[-1]+1,i] = res
                    resf = intp(seg, ptf[seg[0]-1], ptf[seg[0]-1])
                    hfreqIntp[seg[0]:seg[-1]+1,i] = resf
                    resp = intp(seg, ptp[seg[0]-1], ptp[seg[0]-1])
                    hphaseIntp[seg[0]:seg[-1] + 1, i] = resp
                    seg = []
                elif ptNoise[j+1] == 0:
                    if seg[0]==0:
                        res = intp(seg,pt[j+1],pt[j+1])
                        resf = intp(seg,ptf[j+1],ptf[j+1])
                        resp = intp(seg, ptp[j + 1], ptp[j + 1])
                    else:
                        res=intp(seg, pt[seg[0]-1],pt[j+1])
                        resf = intp(seg, ptf[seg[0] - 1], ptf[j + 1])
                        resp = intp(seg, ptp[seg[0] - 1], ptp[j + 1])
                    hmagIntp[seg[0]:seg[-1] + 1, i] = res
                    hfreqIntp[seg[0]:seg[-1] + 1, i] = resf
                    hphaseIntp[seg[0]:seg[-1] + 1, i] = resp

                    seg = []

    return hfreqIntp,hmagIntp,hphaseIntp

def intp(seg,left,right):
    step = np.arange(len(seg)) + 1
    res = left + step*(right-left)/(len(seg)+1)
    return res
###############################################################

########## functions for spectrum modification  ###############
def find_first_frame_phase(ffphase, mode='origin'):
    if mode == 'origin':
        return ffphase,None,None,0
    elif mode == 'random':
        y = 2*np.pi*np.random.rand(ffphase.size)
        err = np.sum((y-ffphase)**2)**(1/2)
        return y,None, None, err
    elif mode == 'linear':
        x = np.arange(len(ffphase))
        slope, intercept, r_value, p_value, std_err = linregress(x, ffphase)
        y = slope*x + intercept
        return y, slope, intercept, std_err

###############################################################

########## functions for sound synthesis from sound controlling parameters  ###############
def non_silence_dtct(fff,minF0,maxF0):
    st = 0
    ed = len(fff)-1
    for i in range(len(fff)):
        if fff[i]<minF0 or fff[i]>maxF0:
            st = i+1
        else:
            break
    for i in range(len(fff)-1,-1,-1):
        if fff[i]<minF0 or fff[i]>maxF0:
            ed = i-1
        else:
            break
    return (st,ed)

def freq_syn_from_para(nH,nF,f0, freqInterval, freqMean, freqVar, freqVarRate, freqSmoothLen):
    hfreqBase = np.arange(nH)*f0*freqInterval + f0
    hfreqBase = np.array([hfreqBase])
    hfreqBase = np.repeat(hfreqBase, nF, axis=0)

    hfreqDist = []
    for i in range(nH):
        mu = freqMean[i]
        sigma = freqVar[i] * freqVarRate

        s = np.random.normal(mu, sigma, nF)
        s = smooth(s, freqSmoothLen)

        hfreqDist.append(s)

    hfreqDist = np.array(hfreqDist).T
    hfreqSyn = hfreqBase + hfreqDist

    return hfreqSyn

def mag_syn_from_para(fs,nH,nF,H,magADSRIndex,magADSRValue,magADSRN):
    hmagSyn = np.zeros((nH,nF))
    t = (np.arange(nF)+0.5) * H / fs
    ts = np.array([t]).T

    for i in range(nH):
        points = np.concatenate((np.array([0]),magADSRIndex[:,i],np.array([nF-1])))
        values = np.concatenate((np.array([-150]),magADSRValue[:,i],np.array([-150])))
        # values = magADSRValue[:, i]
        n = magADSRN[:,i]

        for j in range(5):
            st_ind = points[j]
            ed_ind = points[j + 1]
            st_v = values[j]
            ed_v = values[j+1]

            if st_ind >= ed_ind - 1:
                if st_ind == ed_ind - 1:
                    hmagSyn[i, st_ind] = st_v
            else:
                simCurv = curve(n[j], np.array([[st_v],[ed_v]]), ts[st_ind:ed_ind])
                hmagSyn[i:i + 1, st_ind:ed_ind] = simCurv.T

    hmagSyn[::, -1] = hmagSyn[::, -2]

    return hmagSyn.T

def phase_syn_from_para(fs,H,nH,nF,phaseffSlope,phaseffIntercept,hfreqSyn):
    hphaseSyn = np.zeros((nF,nH))
    hphaseSyn[0,:] = np.arange(nH)*phaseffSlope+phaseffIntercept
    for l in range(1, hphaseSyn.shape[0]):
        hphaseSyn[l, :] = hphaseSyn[l - 1, :] + (np.pi * (hfreqSyn[l - 1, :] + hfreqSyn[l, :]) / fs) * H

    return hphaseSyn

def display_syn(file_path,fs,H,nF,y,yh,yst, hfreqSyn, hmagSyn, hphaseSyn,mode = 1):
    if mode in [1,3]:
        outputFileSines = 'output_sounds/syn_sines.wav'
        outputFileStochastic = 'output_sounds/syn_stochastic.wav'
        outputFile = 'output_sounds/syn.wav'
        UF.wavwrite(yh, fs, outputFileSines)
        UF.wavwrite(yst, fs, outputFileStochastic)
        UF.wavwrite(y, fs, outputFile)

        # UF.wavplay(file_path)
        UF.wavplay(outputFile)

    if mode in [2,3]:
        t = np.arange(nF) * H / fs
        plt.figure()
        plot_spec3d(hfreqSyn, hmagSyn, t, 1)
        plt.figure()
        plot_spec3d(hfreqSyn, hphaseSyn, t, 1)
    return
###########################################################################################


########### save dictionary, read csv/json files ###############################################
def save_dictionary(sdInfo,save_path,fm = 0):
    if fm == 0:  # csv
        path = save_path+'/'+sdInfo['instrument']+'-'+sdInfo['pitch']+'-'+str(sdInfo['index'])+'.csv'
        if os.path.exists(path):
            print('Existed Already: '+path)
        else:
            w = csv.writer(open(path,'w'))
            for key,val in sdInfo.items():
                w.writerow([key,val])
    elif fm == 1:  # json
        path = save_path + '/' + sdInfo['instrument'] + '-' + sdInfo['pitch'] + '-' + str(sdInfo['index']) + '.json'
        if os.path.exists(path):
            print('Existed Already: ' + path)
        else:
            sdInfo['stocEnv'] = sdInfo['stocEnv'].tolist()
            sdInfo['freqMean'] = sdInfo['freqMean'].tolist()
            sdInfo['freqVar'] = sdInfo['freqVar'].tolist()
            sdInfo['magADSRIndex'] = sdInfo['magADSRIndex'].tolist()
            sdInfo['magADSRValue'] = sdInfo['magADSRValue'].tolist()
            sdInfo['magADSRN'] = sdInfo['magADSRN'].tolist()

            with open(path,'w') as f:
                json.dump(sdInfo,f)

    return

def read_features(path,fm = 0):
    # with open(path,mode='r') as file:
    #     reader = csv.reader(file)
    #     sdInfo = {row[0]:row[1] for row in reader}
    if fm == 0: # csv
        df = pd.read_csv(path,header=None,index_col=0)
        sdInfo = df.to_dict()
    elif fm == 1: # json
        f = open(path, 'r')
        sdInfo = json.load(f)

        sdInfo['stocEnv'] = np.array(sdInfo['stocEnv'])
        sdInfo['freqMean'] = np.array(sdInfo['freqMean'])
        sdInfo['freqVar'] = np.array(sdInfo['freqVar'])
        sdInfo['magADSRIndex'] = np.array(sdInfo['magADSRIndex'])
        sdInfo['magADSRValue'] = np.array(sdInfo['magADSRValue'])
        sdInfo['magADSRN'] = np.array(sdInfo['magADSRN'])

    return sdInfo
###########################################################################################

########## utility func ###################################################################
def pitch2freq(pitch):
    if isinstance(pitch, str):
        chroma = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        if pitch[:-1] in chroma:
            pitch = (int(pitch[-1])+1)*12 + chroma.index(pitch[:-1])
        else:
            print('pitch incorrect!')

    return (2**((pitch-69)/12))*440

def herz2rad(f,fs):
    return f/(fs/2)

def dB2abslt(x):
    if isinstance(x,np.ndarray):
        return np.power(10,x/20)
    elif isinstance(x,int) or isinstance(x,float):
        return 10**(x/20)
###########################################################################################

########################## plot spectrogram ###############################################
def plot_spectrogram(sd,fs=44100):
    if isinstance(sd,str):
        (fs, x) = UF.wavread(sd)
    else:
        x = sd

    window = 'hamming'
    M = 4096
    N = 4096
    H = 512
    w = sig.get_window(window,M)
    mX, pX = STFT.stftAnal(x, w, N, H)
    # mX = mX/np.min(mX)

    maxplotfreq = 5000.0
    numFrames = int(mX[:, 0].size)
    frmTime = H * np.arange(numFrames) / float(fs)
    binFreq = fs * np.arange(N * maxplotfreq / fs) / N
    plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:, :int(N * maxplotfreq / fs + 1)]))
    plt.xlabel('time (sec)')
    plt.ylabel('frequency (Hz)')
    plt.title('magnitude spectrogram')
    plt.autoscale(tight=True)
    plt.show()

    return mX,pX,fs,H

###########################################################################################

if __name__ == '__main__':
    ################ test dB2abslt #########################
    print(dB2abslt(np.array([20])))
    ######################################################

    ################ test smooth #########################
    # x = np.arange(10)
    # y = x ** 3 - 4 * x + 1.8
    # y_sm = smooth(y,5)
    # print(y,y_sm)
    ######################################################


    ######## def: pitch2freq ###############
    # print(pitch2freq(69))
    # print(pitch2freq("A4"))
    # print(pitch2freq("C#3"))
    # ########################################

    ######## def: find_first_frame_phase ###############
    # x = np.arange(10)+3
    # y, slope, intercept, err = find_first_frame_phase(x, mode = 'linear')
    # print(y, slope, intercept,err)
    ####################################################

    ######## def: non_silence_dtct ###############
    # x = [0,0,0,2,3,4,5,0,0,5,6]
    # y = non_silence_dtct(x,1,6)
    # print(x[y[0]:y[1]+1])
    ####################################################

    ######## def: non_silence_dtct ###############
    # x = np.array([[0, 0, 0, 2, 3, 4, 5, 0, 0, 5, 6],[0, 0, 0, 2, 3, 4, 5, 0, 0, 5, 6]]).T
    # y = find_freq_failure(x, 1, 6)
    #
    # a,b,c = mag_interpolate(x,y,x,x)
    #
    # print(a)
    # ####################################################

    ################ def: read_features ##################
    # path = 'result/features/flute-A4-0.csv'
    # sdInfo = read_features(path)
    #
    # sdInfo1 = sdInfo.copy()
    # print(sdInfo1)
    # for a in sdInfo1:
    #     print(a, sdInfo1[a],'\n')
    ######################################################

