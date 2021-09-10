import sys,os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../transformations/'))
sys.path.append('/Library/Python/2.7/site-packages/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
import copy

import utilFunctions as UF
import hpsModel as HPS
import harmonicModel as HM
import utilModi as UM
import instrument_feature as INSF
import timbreModi as TM
import find_attack as FA

def harm_ana(file_path):
    (fs, x) = UF.wavread(file_path)

    # parameters
    # window shapex
    window = 'blackmanharris'

    # window size M
    M = 8192

    # fft size
    N = 8192

    # FFT SIZE FOR SYNTHESIS
    Ns = 512

    # HOP SIZE
    H = 128

    # threshold for harmonics in dB
    t = -90

    # min sinusoid duration to be considered as harmonics
    minSineDur = 0.1

    # MAX NUMBER OF HARMONICS
    nH = 20

    # MIN FUNDAMENTAL FREQUENCY
    minf0 = 400

    # MAX FUNDAMENTAL FREQUENCY
    maxf0 = 1200

    # MAX ERROR ACCEPTED IN F0 DETECTION
    f0et = 10

    # MAX ALLOWED DEVIATION OF HARMONIC TRACKS
    harmDevSlope = 0.01

    # DECIMATION FACTOR FOR STOCHASTIC GENERATION
    stocf = 0.1

    w = sig.get_window(window, M)
    hfreq, hmag, hphase, stocEnv = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur,
                                                    Ns, stocf)

    print(hfreq[120:130,::])

    return x, fs, hfreq, hmag, hphase, stocEnv


def find_corr_matrix(hmag):
    hn = hmag.shape[1]
    corrMat = np.zeros((hn,hn))

    st = 0
    ed = hn
    # hmagAve = np.average(hmag,axis=1)
    # print(hmagAve)
    # adsrInd = UM.find_adsr_perc(hmagAve)
    # st = adsrInd[0]
    # ed = adsrInd[-1]
    # print(adsrInd,'\n')


    for h1 in range(hn):
        for h2 in range(h1,hn):
            corr = np.corrcoef(hmag[st:ed,h1], hmag[st:ed,h2])
            corrMat[h1,h2] = corr[0,1]
            corrMat[h2,h1] = corr[1,0]

    return corrMat

def corr_ana(file_path):
    x, fs, hfreq, hmag, hphase, stocEnv = harm_ana(file_path)
    # save_path = '../../../test.xlsx'
    # save_matrix(save_path,hmag,'mag-G4')

    corrMat = find_corr_matrix(hmag)
    return corrMat

def save_matrix(save_path,matrix,sheetName = None, col = None, index = None):
    df = pd.DataFrame(matrix)
    if sheetName == None:
        sheet_name = 'unknown'
    with pd.ExcelWriter(save_path,mode='a') as writer:
        df.to_excel(writer, sheet_name=sheetName)
    return 0

def read_matrix(file_path, sheetName):
    df = pd.read_excel(file_path, sheet_name=sheetName, index_col=0, header=0)
    corrMat = df.to_numpy()
    return corrMat

def find_salient_correlation(corrMat, sheetName):
    corrMatTri = np.triu(corrMat,1)

    # correlation>0.9 to be considered as a similar pair
    # corrSalient = corrMatTri[corrMatTri>0.9]
    # corrSalientInd = np.array(np.where(corrMatTri>0.9)).T
    # corrSalientSort = -np.sort(-corrSalient)
    # corrSalientSortArg = np.argsort(-corrSalient)
    # corrSalientIndSort = corrSalientInd[corrSalientSortArg,::]

    # first 10 pairs to be considered as similar pairs
    corrRsp = np.reshape(corrMatTri,(1,-1))[0]
    corrDes = -np.sort(-corrRsp)
    corrDesArg = np.argsort(-corrRsp)
    corrSalientSort = corrDes[0:10]
    corrSalientIndSort = corrDesArg[0:10]
    corrSalientIndSort = np.array([(corrSalientIndSort/corrMat.shape[0]).astype(int), corrSalientIndSort%corrMat.shape[0]]).T

    print(sheetName, '\n', corrSalientSort,'\n',corrSalientIndSort,'\n')

    return 0

def plot_corr_hm(corr,instmt):

    hn = corr.shape[0]
    label = UM.create_label(hn)
    fig, ax = plt.subplots()
    im = ax.imshow(corr,vmin=0,vmax=1)
    ax.set_title(instmt)
    ax.set_xticks(np.arange(hn))
    ax.set_yticks(np.arange(hn))
    ax.set_xticklabels(label)
    ax.set_yticklabels(label)

    clb = ax.figure.colorbar(im,ax=ax)
    clb.ax.set_ylabel("", rotation=-90, va="bottom")
    plt.show()

def plot_corr_hist(corr,instmt):
    bins = np.arange(21)/20
    plt.hist(np.abs(corr.reshape((1,-1))[0]),bins)
    plt.title(instmt)
    plt.show()

def bubble_sorting(pitch_num, pitch_label, fund_harm, fund_freq):
    for i in range(1, len(pitch_label)):
        for j in range(0, len(pitch_label)-i):
            if pitch_num[j]>pitch_num[j+1]:
                pitch_num[j], pitch_num[j+1] = pitch_num[j+1], pitch_num[j]
                pitch_label[j], pitch_label[j+1] = pitch_label[j+1], pitch_label[j]
                
                temp = copy.deepcopy(fund_harm[:,j])
                fund_harm[:,j] = copy.deepcopy(fund_harm[:,j+1])
                fund_harm[:,j+1] = copy.deepcopy(temp)
                temp = copy.deepcopy(fund_freq[:,j])
                fund_freq[:,j] = copy.deepcopy(fund_freq[:,j+1])
                fund_freq[:,j+1] = copy.deepcopy(temp)

    return pitch_label, fund_harm, fund_freq

if __name__ == '__main__':

    # 2: check correlation between harmonics of a same sound
    # 3: check correlation between fundamental harmonic of different instruments on a same pitch
    # 4: check correlation between fundamental harmonic of the same instrument but on different pitches
    exp = 4

    if exp == 1:
        # instruments: flute, oboe, saxophone, horn, violin, erhu, guitar, harp, piano, organ, trumpet
        # synthesizer: classicPad, pianoStrings, shortWorm, transistorOrgan
        instruments = ['flute', 'oboe', 'saxophone', 'horn', 'violin', 'erhu', 'guitar', 'harp', 'piano', 'organ', 'trumpet']
        ins = ['trumpet']
        # pitches = ['G3','C4','G4','A4','C5','G5','C6','G6']
        # pitches = ['C4', 'G4', 'A4', 'C5', 'G5', 'C6', 'G6']
        pitches = ['C4', 'G4', 'A4', 'C5', 'G5']
        pit = ['G4']
        p3 = ['G3']
        p4 = ['C4','G4','A4']
        p5 = ['C5','G5']
        p6 = ['C6','G6']

        ############ instrument group ################
        # find correlation matrix for each instruments
        # save_path = 'result/result.xlsx'
        # for instmt in instruments:
        #     file_path = '../../sounds/A4/'+ instmt + '-A4.wav'
        #     corrMat = corr_ana(file_path)
        #     save_matrix(save_path,corrMat,sheetName=instmt)

        # find correlation above 0.9 and sort them, different instrument
        # res_path = 'result/result.xlsx'
        # for instmt in instruments:
        #     sheetName = instmt
        #     corrMat = read_matrix(file_path, sheetName)
        #     find_salient_correlation(corrMat, sheetName)

        ##############################################

        ################# single instrument ######################
        # find correlation matrix for flute in each pitch
        # save_path = 'result/violin.xlsx'
        # # for pitch in pitches:
        # for pitch in p5:
        #     file_path = '../../sounds/violin/' + 'violin-' + pitch + '.wav'
        #     if os.path.exists(file_path):
        #         corrMat = corr_ana(file_path)
        #         sheetName = 'violin' + pitch
        #         save_matrix(save_path, corrMat, sheetName=sheetName)
        #         find_salient_correlation(corrMat, sheetName)

        #########################################################

        ################# plot correlation ######################
        save_path = 'result/violin.xlsx'
        # for instmt in ins:
        for p in p5:
            # sheetName = instmt
            sheetName = 'violin'+p
            corr = read_matrix(save_path,sheetName)
            plot_corr_hm(corr,sheetName)
            # plot_corr_hist(corr, sheetName)

        #########################################################

    
    elif exp == 2:
        file_path = '../../sounds/phiharmonia/test/test_len15_A4/'

        for f in os.listdir(file_path):
            if f[-3:] == 'wav':
                print(f)
                instrument, pitch, length, dynamic, articulation = TM.naming_info_phiharmonia(f)

                minf0 = UM.pitch2freq(pitch)-50
                maxf0 = minf0+100 
        
                x, fs, hfreq, hmag, hphase, stocEnv = TM.hps_ana(file_path+f,nH=40,minf0=minf0,maxf0=maxf0,N=4096,M=4096,H=256)
                non_silence = UM.non_silence_dtct(hfreq[:,0], minf0, maxf0)
                corrMat = find_corr_matrix(hmag[non_silence[0]:non_silence[1],:])
                
                # save_path = '../temp.xlsx'
                # save_matrix(save_path, corrMat, sheetName=instrument)
                
                plt.imshow(corrMat, vmin=0, vmax=1)
                plt.title(instrument+' harmonics correlation')
                plt.show()

    
    elif exp == 3:
        file_path = '../../sounds/phiharmonia/test/test_len1_A5/'
        ins_label = []
        flag = 0

        for f in os.listdir(file_path):
            if f[-3:] == 'wav':
                instrument, pitch, length, dynamic, articulation = TM.naming_info_phiharmonia(f)

                minf0 = UM.pitch2freq(pitch)-50
                maxf0 = minf0+100 
        
                x, fs, hfreq, hmag, hphase, stocEnv = TM.hps_ana(file_path+f,nH=40,minf0=minf0,maxf0=maxf0,N=4096,M=4096,H=256)
                non_silence = UM.non_silence_dtct(hfreq[:,0], minf0, maxf0)
                hmag = hmag[non_silence[0]:non_silence[1],0]
                hfreq = hfreq[non_silence[0]:non_silence[1],0]
                
                ins_label.append(instrument)
                harm0 = np.zeros((100,1))
                freq0 = np.zeros((100,1))
                nF = hmag.size
                for i in range(100):
                    ind = int(i/100*nF)
                    harm0[i] = hmag[ind]
                    freq0[i] = hfreq[ind]

                if flag == 0:
                    fund_harm = harm0
                    fund_freq = freq0
                    flag = 1
                else:
                    fund_harm = np.concatenate((fund_harm, harm0),axis=1)
                    fund_freq = np.concatenate((fund_freq, freq0),axis=1)

        instrumentOrder = ['flute','oboe','clarinet','saxophone','french-horn','trumpet','violin','cello']
        ins_num = []
        for i in ins_label:
            ins_num.append(instrumentOrder.index(i))
        ins_label, fund_harm, fund_freq = bubble_sorting(ins_num, ins_label, fund_harm, fund_freq)

        # fund_harm = np.array(fund_harm)
        # print(fund_harm.shape)
        corrMat = find_corr_matrix(fund_harm)
       
        num = len(ins_label)
        fig, ax = plt.subplots(1,1)
        # img = ax.imshow(corrMat, vmin=0, vmax=1)
        img = ax.imshow(corrMat)
        ax.set_xticks(np.arange(num))
        ax.set_xticklabels(ins_label)
        ax.set_yticks(np.arange(num))
        ax.set_yticklabels(ins_label)
        plt.title('correlation beteewn fundamental harmonics on '+pitch)
        fig.colorbar(img)
        plt.show()
        
        fund_hfreq = np.repeat(np.array([np.arange(num)]).T, 100, axis=1)
        UM.plot_spec3d(fund_hfreq.T, fund_harm, np.arange(100))


    elif exp == 4:
        file_path = '../../sounds/phiharmonia/test/test_violin/'
        pitch_label = []
        flag = 0

        for f in os.listdir(file_path):
            if f[-3:] == 'wav':
                instrument, pitch, length, dynamic, articulation = TM.naming_info_phiharmonia(f)

                minf0 = UM.pitch2freq(pitch)-50
                maxf0 = minf0+100 

                x, fs, hfreq, hmag, hphase, stocEnv = TM.hps_ana(file_path+f,nH=40,minf0=minf0,maxf0=maxf0,N=4096,M=4096,H=256)
                non_silence = UM.non_silence_dtct(hfreq[:,0], minf0, maxf0)
                hmag = hmag[non_silence[0]:non_silence[1],0]
                # hmag = FA.find_spline_harmonic(hmag, lam=10000)
                hfreq = hfreq[non_silence[0]:non_silence[1],0]

                pitch_label.append(pitch)
                harm0 = np.zeros((100,1))
                freq0 = np.zeros((100,1))
                nF = hmag.size
                for i in range(100):
                    ind = int(i/100*nF)
                    harm0[i] = hmag[ind]
                    freq0[i] = hfreq[ind]

                if flag == 0:
                    fund_harm = harm0
                    fund_freq = freq0
                    flag = 1
                else:
                    fund_harm = np.concatenate((fund_harm, harm0),axis=1)
                    fund_freq = np.concatenate((fund_freq, freq0),axis=1)

        # sort according to the pitches
        chroma = ['C','Cs','D','Ds','E','F','Fs','G','Gs','A','As','B']
        pitch_num = []
        for i in range(len(pitch_label)):
            pitch_num.append((int(pitch_label[i][-1])+1)*12 + chroma.index(pitch_label[i][:-1]))
        pitch_label, fund_harm, fund_freq = bubble_sorting(pitch_num, pitch_label, fund_harm, fund_freq)

        print()
        # fund_harm = np.array(fund_harm)
        corrMat = find_corr_matrix(fund_harm)
        print(corrMat)
        
        fig, ax = plt.subplots(1,1)
        img = ax.imshow(corrMat, vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(pitch_label)))
        ax.set_xticklabels(pitch_label)
        ax.set_yticks(np.arange(len(pitch_label)))
        ax.set_yticklabels(pitch_label)
        plt.title('correlation beteewn fundamental harmonics for '+instrument)
        fig.colorbar(img)
        plt.show()
        
        UM.plot_spec3d(fund_freq, fund_harm, np.arange(100))
        plt.subplot(1,2,1)
        plt.plot(fund_harm[:,0])
        plt.title('Harmonic # 1')
        plt.subplot(1,2,2)
        plt.plot(fund_harm[:,1])
        plt.title('Harmonic # 2')
        plt.show()

