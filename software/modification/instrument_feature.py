import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../transformations/'))
sys.path.append('/Library/Python/2.7/site-packages/')

import numpy as np
from random import uniform
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error,classification_report,plot_confusion_matrix,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.inspection import permutation_importance
from sklearn.cluster import k_means
from sklearn.model_selection import train_test_split
import scipy.signal as sig
from joblib import dump,load


import utilModi as UM
import utilFunctions as UF
import timbreModi as TM
import harmonicModel as HM

def json2vector(f,raw=0,nH=40):
    if isinstance(f, str) and os.path.exists(f):
        sdInfo = UM.read_features(f,fm=1)
    elif isinstance(f, str) and not os.path.exists(f):
        print('file '+f+' not found!')
        return
    else:
        sdInfo = f
        
    harmClct = nH
    if sdInfo['nH'] >= harmClct:
        attackTimeAbs = (sdInfo['magADSRIndex'][1,:] - sdInfo['magADSRIndex'][0,:])*sdInfo['hopSize']/sdInfo['fs']
        #releaseTimeAbs = (sdInfo['magADSRIndex'][3,:] - sdInfo['magADSRIndex'][2,:])*sdInfo['hopSize']/sdInfo['fs']
        releaseTimeRlt = (sdInfo['magADSRIndex'][3, :] - sdInfo['magADSRIndex'][2, :])/(sdInfo['magADSRIndex'][3, :] - sdInfo['magADSRIndex'][0, :])

        if raw == 0: # return normalized mag value
            magADSRValue = dB2amp(sdInfo['magADSRValue']) # use real amplitude instead of dB scale
            magADSRValueNorm = magADSRValue/np.repeat(magADSRValue[1:2,:],4,axis=0)
            # actually the second point is not always the max, but when it is not, it's always pretty close to the max(usually the third point)
            magADSRValueNorm = np.concatenate((magADSRValueNorm[0:1,:],magADSRValueNorm[2:,:]))[:,:harmClct]
            magADSRValueMax = (magADSRValue[1,:]/np.sum(magADSRValue[1,:]))[:harmClct] # to avoid the interference of dynamics

            ft = [sdInfo['f0']] + sdInfo['freqMean'][:harmClct].tolist() + sdInfo['freqVar'][:harmClct].tolist() + attackTimeAbs[:harmClct].tolist()\
                 + releaseTimeRlt[:harmClct].tolist() + np.reshape(magADSRValueNorm,-1).tolist() + magADSRValueMax.tolist()\
                 + np.reshape(sdInfo['magADSRN'][1:,:harmClct],-1).tolist() + [sdInfo['phaseffSlope'],sdInfo['phaseffIntercept']]

        elif raw == 1:
            magADSRValue = dB2amp(sdInfo['magADSRValue'])
            ft = [sdInfo['f0']] + sdInfo['freqMean'][:harmClct].tolist() + sdInfo['freqVar'][:harmClct].tolist() + attackTimeAbs[:harmClct].tolist() \
                 + releaseTimeRlt[:harmClct].tolist() + magADSRValue.tolist() \
                 + np.reshape(sdInfo['magADSRN'][1:, :harmClct], -1).tolist() + [sdInfo['phaseffSlope'],sdInfo['phaseffIntercept']]

        else:
            print('wrong raw parameter, returning raw date.')
            magADSRValue = dB2amp(sdInfo['magADSRValue'])
            ft = [sdInfo['f0']] + sdInfo['freqMean'][:harmClct].tolist() + sdInfo['freqVar'][:harmClct].tolist() + attackTimeAbs[:harmClct].tolist() \
                 + releaseTimeRlt[:harmClct].tolist() + magADSRValue.tolist() \
                 + np.reshape(sdInfo['magADSRN'][1:, :harmClct], -1).tolist() + [sdInfo['phaseffSlope'],sdInfo['phaseffIntercept']]
        
        return ft
    else:
        print("Need"+ str(harmClct)+ "harmonics, only ",sdInfo['nH']," collected!")
        return


def get_feature_list(dir,fts,raw,nH,minPitch=0,maxPitch=180,minFreq=20,maxFreq=48000):
    newDir = dir
    if os.path.isfile(dir) and dir[-4:] == 'json':
        ft = json2vector(dir,raw,nH)
        if ft[0] >= minFreq and ft[0] <= maxFreq:
            if np.isnan(np.sum(ft)):
                print(dir+" has NaN value(s)!")
            else:
                fts.append(ft)
        
    elif os.path.isdir(dir):
        for f in os.listdir(dir):
            newDir = os.path.join(dir,f)
            fts = get_feature_list(newDir,fts,raw,nH)
    return fts

def feature_vectorization(files_path,raw = 0,nH = 40, minPitch=0, maxPitch=180, minFreq=20, maxFreq=48000):
    fts = []
    fts = get_feature_list(files_path,fts,raw,nH,minPitch,maxPitch,minFreq,maxFreq)
    return np.array(fts).T

def feature_vectorization_devide_set(files_path,raw=0,nH = 40, trainSetRate = 0.8, minPitch=0, maxPitch=180, minFreq=20, maxFreq=48000):
    fts_train = []
    fts_test = []

    minFreq = max(minFreq, UM.pitch2freq(minPitch))
    maxFreq  = min(maxFreq, UM.pitch2freq(maxPitch))

    for f in os.listdir(files_path):
        if f[-4:] == 'json':
            ft = json2vector(files_path+f,raw,nH)
            if ft[0] >= minFreq and ft[0] <= maxFreq:
                if np.isnan(np.sum(ft)):
                    print(f+" has NaN value(s)!")
                else:
                    if uniform(0,1) < trainSetRate:
                        fts_train.append(ft)
                    else:
                        fts_test.append(ft)

    return np.array(fts_train).T, np.array(fts_test).T

def vector2dict(x,Y,nF=1000,silenceSt = 10, silenceEd = 10, meanMax = -100,nH = 40):
    sdInfo = {'instrument':'synthesis',
              'pitch':'',
              'source':'',
              'index':'111',
              'nH':nH,
              'nF':nF,
              'FFTLenAna':8192,
              'FFTLenSyn':512,
              'hopSize':256,
              'fs':44100,
              'stocEnv': -50*np.ones((nF,12)), # no noise
              'f0': x[0],
              'freqInterval':1,
              'freqVarRate':0,
              'freqSmoothLen':31}

    cursor = 0
    sdInfo['freqMean'] = Y[cursor:cursor+nH]
    cursor += nH
    sdInfo['freqVar'] = Y[cursor:cursor+nH]
    cursor += nH

    # recover index
    magRiseTimeAbs = np.array(Y[cursor:cursor+nH])/sdInfo['hopSize']*sdInfo['fs']
    cursor += nH
    magReleaseTimeRlt = np.array(Y[cursor:cursor+nH])
    cursor += nH
    slSt = silenceSt * np.ones(nH)
    slEd = silenceSt * np.ones(nH)
    magADSRIndex = np.vstack((slSt,slSt+magRiseTimeAbs,nF-slEd-magReleaseTimeRlt*(nF-slSt-slEd),nF-slEd)).astype(int)
    sdInfo['magADSRIndex'] = magADSRIndex

    # recover amplitude
    magADSRValue = np.reshape(Y[cursor:cursor + 3*nH],(3,-1))
    cursor += 3*nH

    magADSRValueMax = np.array([Y[cursor:cursor+nH]])*dB2amp(meanMax)*nH
    cursor += nH
    # print(magADSRValue,magADSRValueMax)
    magADSRN = np.vstack((np.ones(nH),np.reshape(Y[cursor:cursor+4*nH],(4,-1))))
    cursor += 4*nH
    magADSRValue = np.vstack((magADSRValue[0,:]*magADSRValueMax,magADSRValueMax,magADSRValue[1,:]*magADSRValueMax,magADSRValue[2,:]*magADSRValueMax))
    sdInfo['magADSRValue'] = amp2dB(magADSRValue)
    sdInfo['magADSRN'] = magADSRN

    # recover phase
    sdInfo['phaseffSlope'] = Y[cursor]
    cursor += 1
    sdInfo['phaseffIntercept'] = Y[cursor]

    return sdInfo

def dB2amp(X):
    return 10**(X/20)

def amp2dB(X):
    return 20*np.log10(X)

# typically for freqMean
def find_non_nan_ind(Y):
    nH = (Y.shape[1]-2)//13
    Yfreqm = Y[:,0:nH]
    nonNanInd = (Yfreqm == Yfreqm).astype(int)
    return nonNanInd

# regression for all features for one instrument
def feature_analysis(ftMatTrain,ftMatTest,Xpredict,display,predict=0):
    feature_name = ['freq Mean','freq Variance','absolute rise time','relative release time','mag value pt1/pt2','mag value pt3/pt2',
                    'mag value pt4/pt2','mag pt2 portion','mag N 1','mag N 2','mag N 3','mag N 4','mag N 5','first frame phase slope',
                    'first frame phase intercept']

    XTrain = ftMatTrain[0:1,:].T
    YTrain = ftMatTrain[1:,:].T
    XTest = ftMatTest[0:1,:].T
    YTest = ftMatTest[1:,:].T

    nH = (YTrain.shape[1] - 2) // 13
    ftNum = YTrain.shape[1]

    Ypredict = np.zeros((len(Xpredict),ftNum))
    Xpredict = np.array([Xpredict]).T

    nonNanIndTrain = find_non_nan_ind(YTrain)
    nonNanIndTest = find_non_nan_ind(YTest)

    for i in range(ftNum):
        showFg = ((i%nH == 0 & i<ftNum-2) | (i == ftNum - 1))and display == 1

        yTrain = YTrain[:,i:i+1]
        yTest = YTest[:,i:i+1]

        if i < ftNum-2:
            indTrain = nonNanIndTrain[:,i % nH:i % nH + 1]
            indTest = nonNanIndTest[:, i % nH:i % nH + 1]
            xNewTrain = np.array([XTrain[np.where(indTrain)]]).T
            yNewTrain = np.array([yTrain[np.where(indTrain)]]).T
            xNewTest = np.array([XTest[np.where(indTest)]]).T
            yNewTest = np.array([yTest[np.where(indTest)]]).T

        else:
            xNewTrain = XTrain
            yNewTrain = yTrain
            xNewTest = XTest
            yNewTest = yTest

        # plotting
        xPlot = np.array([np.linspace(0,2499,2500)]).T

        allRegMethod = ['linear', 'poly_2', 'poly_3', 'linear_l2', 'poly_2_l2', 'poly_3_l2']  # 'all','all_poly','all_ridge'
        regMethod = 'none'
        regParas = []
        allErrTrain = np.zeros((len(allRegMethod), ftNum))
        allErrTest = np.zeros((len(allRegMethod), ftNum))

        # implement regression
        if regMethod not in ['all','all_poly','all_ridge','none'] and regMethod not in allRegMethod:
            print('Regression method', regMethod, ' not supported, use default linear regression. Feature number:', i)
            regMethod = 'linear'

        if regMethod in ['linear','all','all_poly']:
            reg1 = LinearRegression().fit(xNewTrain,yNewTrain)
            errTrain = mean_squared_error(yNewTrain,reg1.predict(xNewTrain))
            errTest = mean_squared_error(yNewTest,reg1.predict(xNewTest))
            # print(reg.coef_,reg.intercept_,err)
            allErrTrain[allRegMethod.index('linear'),i] = errTrain
            allErrTest[allRegMethod.index('linear'), i] = errTest

            if showFg:
                plt.plot(xPlot,reg1.predict(xPlot),'k')

        if regMethod in ['poly_2','all','all_poly']:
            poly2 = PolynomialFeatures(degree=2)
            xPoly = poly2.fit_transform(xNewTrain)
            reg2 = LinearRegression().fit(xPoly,yNewTrain)
            errTrain = mean_squared_error(yNewTrain,reg2.predict(xPoly))
            errTest = mean_squared_error(yNewTest, reg2.predict(poly2.fit_transform(xNewTest)))
            # print(reg.coef_,reg.intercept_,err)
            allErrTrain[allRegMethod.index('poly_2'), i] = errTrain
            allErrTest[allRegMethod.index('poly_2'), i] = errTest

            if showFg:
                plt.plot(xPlot, reg2.predict(poly2.fit_transform(xPlot)), 'b')


        if regMethod in ['poly_3','all','all_poly']:
            poly3 = PolynomialFeatures(degree=3)
            xPoly = poly3.fit_transform(xNewTrain)
            reg3 = LinearRegression().fit(xPoly, yNewTrain)
            errTrain = mean_squared_error(yNewTrain, reg3.predict(xPoly))
            errTest = mean_squared_error(yNewTest, reg3.predict(poly3.fit_transform(xNewTest)))
            # print(reg.coef_, reg.intercept_, err)
            allErrTrain[allRegMethod.index('poly_3'), i] = errTrain
            allErrTest[allRegMethod.index('poly_3'), i] = errTest

            if showFg:
                plt.plot(xPlot, reg3.predict(poly3.fit_transform(xPlot)), 'y')


        if regMethod in ['linear_l2','all','all_ridge']:
            reg4 = Ridge(alpha=1).fit(xNewTrain,yNewTrain)
            errTrain = mean_squared_error(yNewTrain, reg4.predict(xNewTrain))
            errTest = mean_squared_error(yNewTest, reg4.predict(xNewTest))
            allErrTrain[allRegMethod.index('linear_l2'), i] = errTrain
            allErrTest[allRegMethod.index('linear_l2'), i] = errTest

            if showFg:
                plt.plot(xPlot, reg4.predict(xPlot), 'r')

        if regMethod in ['poly_2_l2','all','all_ridge']:
            poly2 = PolynomialFeatures(degree=2)
            xPoly = poly2.fit_transform(xNewTrain)
            reg5 = Ridge(alpha=1).fit(xPoly,yNewTrain)
            errTrain = mean_squared_error(yNewTrain, reg5.predict(xPoly))
            errTest = mean_squared_error(yNewTest, reg5.predict(poly2.fit_transform(xNewTest)))
            allErrTrain[allRegMethod.index('poly_2_l2'), i] = errTrain
            allErrTest[allRegMethod.index('poly_2_l2'), i] = errTest

            if showFg:
                plt.plot(xPlot, reg5.predict(poly2.fit_transform(xPlot)), 'g')

        if regMethod in ['poly_3_l2','all','all_ridge']:
            poly3 = PolynomialFeatures(degree=3)
            xPoly = poly3.fit_transform(xNewTrain)
            reg6 = Ridge(alpha=1).fit(xPoly,yNewTrain)
            errTrain = mean_squared_error(yNewTrain, reg6.predict(xPoly))
            errTest = mean_squared_error(yNewTest, reg6.predict(poly3.fit_transform(xNewTest)))
            allErrTrain[allRegMethod.index('poly_3_l2'), i] = errTrain
            allErrTest[allRegMethod.index('poly_3_l2'), i] = errTest

            if showFg:
                plt.plot(xPlot, reg6.predict(poly3.fit_transform(xPlot)), 'c')

        if i == 280:
            yMagFirstHarmTrain = yNewTrain
            yMagFirstHarmTest = yNewTest
        # if i>=13*nH and display == 1:
        # if i>=280 and i<320 and display == 1:
        if i%nH in [0,1,2,3,9] and display == 1:
        #     plt.plot(xNewTrain,yNewTrain/np.array([yMagFirstHarmTrain[np.where(indTrain)]]).T,'bx')
        #     plt.plot(xNewTest, yNewTest/np.array([yMagFirstHarmTest[np.where(indTest)]]).T, 'rx')
            plt.plot(xNewTrain, yNewTrain, 'bx')
            plt.plot(xNewTest, yNewTest, 'rx')
            plt.title(feature_name[i//nH]+' #'+str(i%nH))
            plt.show()


        # calculate value for the X to be predicted
        if predict == 1:
            regs = [reg1,reg2,reg3,reg4,reg5,reg6]
            model = np.argmin(allErrTest[:,i])
            # print(model)
            if model in [1,4]:
                XpredictNew = poly2.fit_transform(Xpredict)
            elif model in [2,5]:
                XpredictNew = poly3.fit_transform(Xpredict)
            else:
                XpredictNew = Xpredict
            Ypredict[:,i:i+1] = regs[model].predict(XpredictNew)


    return allErrTrain, allErrTest, Ypredict

# select the maximum magnitude keypoint
def get_second_magkp_ratio(ftMat):
    f0 = ftMat[0:1,:]
    ft = ftMat[1:,:]
    nH = (ft.shape[1] - 2) // 13
    ftNum = ft.shape[1]

    magkp2 = ft[5*nH:6*nH,:]
    magkp2 = magkp2/np.repeat(magkp2[0:1,:],nH,axis=0)
    return magkp2

# get features as inputs and instrument as labels for mixed instrument
# given instrument list, return training and testing labels and inputs matrix
def get_mixed_ins_data(instruments, nH = 40, source = 0, minPitch=0, maxPitch=180, minFreq=20, maxFreq=48000):
    for ins in instruments:
        if source == 0 or source == 'NSynth':
            train_path = 'result/features/' + ins + '_acoustic/' + ins + '_acoustic_valid'
            test_path = 'result/features/' + ins + '_acoustic/' + ins + '_acoustic_test'
            ftMatTrain = feature_vectorization(train_path, nH=nH, minPitch=minPitch, maxPitch=maxPitch, minFreq=minFreq, maxFreq=maxFreq)
            ftMatTest = feature_vectorization(test_path, nH=nH, minPitch=minPitch, maxPitch=maxPitch, minFreq=minFreq, maxFreq=maxFreq)
        elif source == 1 or source == 'phiharmonia':
            files_path = 'result/features/phiharmonia/' + ins + '/'
            ftMatTrain, ftMatTest = feature_vectorization_devide_set(files_path, nH = nH, trainSetRate= 0.8, minPitch=minPitch, maxPitch=maxPitch, minFreq=minFreq, maxFreq=maxFreq)
       
        # tst
        # print(ftMatTrain.shape)

        ftNum, dataNumTrain = ftMatTrain.shape
        _, dataNumTest = ftMatTest.shape
        ftNum = ftNum - 1

        if instruments.index(ins) == 0:
            labelTrain = np.zeros((1, dataNumTrain))
            labelTest = np.zeros((1, dataNumTest))
            inputTrain = ftMatTrain[1:, :]
            inputTest = ftMatTest[1:, :]
            f0Train = ftMatTrain[0:1,:]
            f0Test = ftMatTest[0:1,:]

        else:
            labelTrain = np.column_stack((labelTrain, np.ones((1, dataNumTrain)) * instruments.index(ins)))
            labelTest = np.column_stack((labelTest, np.ones((1, dataNumTest)) * instruments.index(ins)))
            inputTrain = np.column_stack((inputTrain, ftMatTrain[1:, :]))
            inputTest = np.column_stack((inputTest, ftMatTest[1:, :]))
            f0Train = np.column_stack((f0Train, ftMatTrain[0:1, :]))
            f0Test = np.column_stack((f0Test, ftMatTrain[0:1, :]))

        print(ins,' data collecting finished.',labelTest.shape)

    return inputTrain,labelTrain.astype(int),f0Train,inputTest,labelTest.astype(int),f0Test

def timbre_classification(instruments, inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test, display = 1, class_label=None, display_label=None, return_model=0):
    # reassign the labels
    if class_label !=  None:
        class_label = np.array(class_label)
        labelTrain = class_label[labelTrain]
        labelTest = class_label[labelTest]

    # clf = svm.SVC(decision_function_shape='ovr',probability=False)
    clf = RandomForestClassifier(n_estimators=100,random_state=0,oob_score=True)
    clf.fit(inputTrain.T,labelTrain[0,:])
    YTrain = clf.predict(inputTrain.T).astype(int)
    YTest = clf.predict(inputTest.T).astype(int)

    trainScore = clf.score(inputTrain.T, labelTrain[0,:])
    testScore = clf.score(inputTest.T, labelTest[0,:])

    print("Train", trainScore)
    print("Test", testScore)
    print("OOB score",clf.oob_score_)

    # print(classification_report(labelTrain[0,:],YTrain,instruments))
    # print(classification_report(labelTest[0, :],YTest, instruments))
    if display == 1: 
        if display_label == None:
            display_labels = instruments
        else:
            display_labels = display_label

        disp = plot_confusion_matrix(clf, inputTrain.T, labelTrain[0, :], display_labels=display_labels, normalize='true')
        plt.show()
        # print("Train Confusion matrix:\n%s" % disp.confusion_matrix)
        disp = plot_confusion_matrix(clf,inputTest.T,labelTest[0, :], display_labels=display_labels, normalize = 'true')
        # print("Test Confusion matrix:\n%s" % disp.confusion_matrix)
        plt.show()

    # find features' importance by calculating Gini
    importance_gini = clf.feature_importances_
    plt.plot(importance_gini)
    plt.title('feature importance, VIM_Gini')
    plt.show()

    # find features' importance by permutation
    # importance_perm = permutation_importance(clf, inputTrain.T, labelTrain[0,:], n_repeats=10, random_state=1, n_jobs=2)
    # plt.plot(importance_perm.importances_mean)
    # plt.title('feature importance, permutation')
    # print('permutation importance mean:',importance_perm.importances_mean)
    # print('permutation importance std:',importance_perm.importances_std)

    # check where the errors fall
    # reedErr = f0Test[0,np.where(YTest<labelTest[0, :])][0,:]
    # print(reedErr)
    # reedOri = f0Test[0,np.where(labelTest[0,:] == 1)][0,:]
    # print(reedOri.shape)
    # # bins = np.arange(25)*100
    # plt.subplot(2,1,1)
    # plt.hist(reedOri,bins=100)
    # plt.title('distribution of frequency for reed test')
    # plt.subplot(2,1,2)
    # plt.hist(reedErr,bins=100)
    # plt.title('distribution of frequency for reed being classfied as brass in test')
    # plt.show()
    
    if return_model == 1:
        return (trainScore, testScore), clf
    return (trainScore, testScore)

def feature_PCA(instruments, inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test, n_components=10, display = 1):
    n_ft, n_num_train = inputTrain.shape
    _,n_num_test = inputTest.shape

    # normalization (train+test)
    normMax = np.array([np.max(np.abs(inputTrain),axis=1),np.max(np.abs(inputTest),axis=1)])
    inputMaxRep = np.repeat(np.array([np.max(normMax,axis=0)]).T,n_num_train,axis=1)
    inputTrainNorm = inputTrain/inputMaxRep
    inputMaxRep = np.repeat(np.array([np.max(normMax, axis=0)]).T, n_num_test, axis=1)
    inputTestNorm = inputTest/inputMaxRep

    pca = PCA(n_components=n_components)
    pca.fit(inputTrainNorm.T)

    # print some parameters
    # covar = pca.get_covariance()
    # print(pca.explained_variance_ratio_)

    inputTrainDecomp = pca.transform(inputTrainNorm.T)
    inputTestDecomp = pca.transform(inputTestNorm.T)
    _, n_ft_new = inputTrainDecomp.shape
    corel = np.corrcoef(inputTrainNorm.T,inputTrainDecomp,rowvar=False)
    corr = corel[0:n_ft,n_ft:]
    # UM.save_matrix('../../../test.xlsx',corel,sheetName='PCA_norm_correlarion_coefficient')
    
    # print('correlation matrix:',corr)
    
    if display == 1:
        fig, ax = plt.subplots()
        im = ax.imshow(np.abs(corr),vmin=0,vmax=1)
        clb = ax.figure.colorbar(im, ax=ax)
        clb.ax.set_ylabel("", rotation=-90, va="bottom")
        plt.show()

    # do classification
    return timbre_classification(instruments, inputTrainDecomp.T, labelTrain, f0Train, inputTestDecomp.T, labelTest, f0Test)

def feature_LDA(instruments, inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test):
    n_ft, n_num_train = inputTrain.shape
    _, n_num_test = inputTest.shape

    # normalization (train+test)
    normMax = np.array([np.max(np.abs(inputTrain), axis=1), np.max(np.abs(inputTest), axis=1)])
    inputMaxRep = np.repeat(np.array([np.max(normMax, axis=0)]).T, n_num_train, axis=1)
    inputTrainNorm = inputTrain / inputMaxRep
    inputMaxRep = np.repeat(np.array([np.max(normMax, axis=0)]).T, n_num_test, axis=1)
    inputTestNorm = inputTest / inputMaxRep

    # generate a classifier
    clf = LinearDiscriminantAnalysis(n_components=3)
    clf.fit(inputTrainNorm.T, labelTrain[0,:])
    inputTrainDecomp = clf.transform(inputTrainNorm.T)
    # inputTestDecomp = clf.transform(inputTestNorm.T)
    _, n_ft_new = inputTrainDecomp.shape
    corel = np.corrcoef(inputTrainNorm.T,inputTrainDecomp,rowvar=False)
    corr = corel[0:n_ft,n_ft:]
    # UM.save_matrix('../../../test.xlsx',corel,sheetName='LDA_norm_correlarion_coefficient')

    # print correlation mat
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(corr),vmin=0,vmax=1)
    clb = ax.figure.colorbar(im, ax=ax)
    clb.ax.set_ylabel("", rotation=-90, va="bottom")
    plt.show()

    # print accuracy
    trainScore = clf.score(inputTrainNorm.T, labelTrain[0,:])
    testScore = clf.score(inputTestNorm.T, labelTest[0,:])

    print("Train", trainScore)
    print("Test", testScore)

    # print confusion matrix
    disp = plot_confusion_matrix(clf, inputTrainNorm.T, labelTrain[0, :])
    plt.show()
    # print("Train Confusion matrix:\n%s" % disp.confusion_matrix)
    disp = plot_confusion_matrix(clf, inputTestNorm.T, labelTest[0, :])
    plt.show()

    return trainScore, testScore

def NaN_finder(mat):
    return np.where(np.isnan(mat))

def harmonicSelection(inputTrain, inputTest, harmSlct):
    ft_prsv = np.repeat(np.array([np.arange(12)*40]), harmSlct.size, axis=0) + np.repeat(np.array([harmSlct]).T, 12, axis=1)
    ft_prsv = np.reshape(ft_prsv, (1,-1), order='F')[0,:]
    newTrain = inputTrain[ft_prsv,:]
    newTest = inputTest[ft_prsv,:]
    return newTrain, newTest

def save_model(clf, file_name):
    dump(clf, file_name)

    # tst
    print('joblib saved!')


if __name__ == '__main__':
    
    # 0: function testing
    # 1: regression for all features for one instrument
    # 2: max magnitude key point clustering for one instrument
    # 3: classification for different instruments
    # 4: listen to single feature modification
    # 5: do some clusterings
    # 6: sound morphing
    exp = 3

    if exp == 1: # regression for all features for one instrument
        train_path = 'result/features/brass_acoustic/brass_acoustic_valid'
        ftMatTrain = (train_path)
        test_path = 'result/features/brass_acoustic/brass_acoustic_test'
        ftMatTest = feature_vectorization(test_path)
        Xpredict = np.array([])
        allErrTrain, allErrTest, Ypredict = feature_analysis(ftMatTrain,ftMatTest,Xpredict,display=1,predict=0)

        # synthesis Ypredict
        for i in range(Xpredict.size):
            sdInfo = vector2dict([Xpredict[i]], Ypredict[i,:], nF=1000, silenceSt=10, silenceEd=10, meanMax=-100)
            y, yh, yst, hfreqSyn, hmagSyn, hphaseSyn = TM.sound_syn_from_para(sdInfo)

            outputFile = 'output_sounds/syn.wav'
            UF.wavwrite(y, sdInfo['fs'], outputFile)
            UF.wavplay('../../sounds/flute/flute-A4.wav')
            UF.wavplay(outputFile)

            t = np.arange(sdInfo['nF']) * sdInfo['hopSize'] / sdInfo['fs']
            plt.figure()
            UM.plot_spec3d(hfreqSyn[:,0:10], hmagSyn[:,0:10], t, 1)

    elif exp == 2: # max magnitude key point clustering for one instrument
        train_path = 'result/features/brass_acoustic/brass_acoustic_valid'
        ftMatTrain = feature_vectorization(train_path,raw=1)
        test_path = 'result/features/brass_acoustic/brass_acoustic_test'
        ftMatTest = feature_vectorization(test_path,raw=1)

        magkp2RatioTrain = get_second_magkp_ratio(ftMatTrain)
        magkp2RatioTest = get_second_magkp_ratio(ftMatTest)

    elif exp == 3:
        instruments_test = ['flute','string','brass','reed']
        instruments = ['flute','oboe','clarinet','saxophone','french horn','trumpet','violin','cello']
        # instruments = ['flute','cello','trumpet','oboe','clarinet']
        class_label = [0,1,2,3,3]
        display_label = ['flute','string','brass','reed']
        
        # harmonics numbers for training
        # nH_list = range(1,41)
        nH_list = [40]
        # decomp_list = range(10,490,10)
        decomp_list = [10]
        maxFreq = 550

        trainAccuracy = []
        testAccuracy = []

        for nH in nH_list:
            print('Harmonics Number:',nH)
            # get inputs and labels
            inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test = get_mixed_ins_data(instruments,nH=nH, source=1,maxFreq = maxFreq) # source0: Synth, source1: phiharmonia
            # inputTrain = np.column_stack((inputTrain, inputTest))
            # labelTrain = np.column_stack((labelTrain, labelTest))
           
            # inputTest1, labelTest1, f0Test1, inputTest2, labelTest2, f0Test2 = get_mixed_ins_data(instruments_test,nH=nH, source=0,maxFreq = maxFreq) # source0: Synth, source1: phiharmonia
            # inputTrain = np.column_stack((inputTrain, inputTest1))
            # labelTrain = np.column_stack((labelTrain, labelTest1))
            # inputTest = np.column_stack((inputTest, inputTest2))
            # labelTest = np.column_stack((labelTest, labelTest2))
            
            # harmSlct = np.array([0,1,2,3,4,5])
            # harmSlct = np.arange(10)*2 # odd harms
            # harmSlct = np.arange(10)*2+1 # even harms
            #inputTrain, inputTest = harmonicSelection(inputTrain, inputTest, harmSlct)

            # select features based on timbre space: rise time, and mag values
            # inputTrain = np.vstack((inputTrain[nH*2:nH*3,:],inputTrain[nH*4:nH*8,:]))
            # inputTest = np.vstack((inputTest[nH*2:nH*3,:],inputTest[nH*4:nH*8,:]))

            # select the complementary of timbre space
            # inputTrain = np.vstack((inputTrain[:nH*2,:],inputTrain[nH*3:nH*4,:],inputTrain[nH*8:,:]))
            # inputTest = np.vstack((inputTest[:nH*2, :], inputTest[nH*3:nH*4, :], inputTest[nH*8:, :]))
            
            # inputTrain = inputTrain[nH*4:nH*8,:]
            # inputTest = inputTest[nH*4:nH*8,:]

            ## tst
            # print('inputTrain NaN index:')
            # print(NaN_finder(inputTrain)[0])
            # print(NaN_finder(inputTrain)[1])
            # print('inputTest NaN index:')
            # print(NaN_finder(inputTest)[0])
            # print(NaN_finder(inputTest)[1])
            # # UM.save_matrix('../temp.xlsx', NaN_finder(inputTrain)[0], sheetName='nan_train')
            # UM.save_matrix('../temp.xlsx',inputTest[8:11,:],sheetName='inputTest_8-10')
            # UM.save_matrix('../temp.xlsx', inputTest[48:51,:],sheetName='inputTest_48-51')

            # print('train feature dim, train data num', inputTrain.shape)
            # print('test feature dim, test data num', inputTest.shape)        

            # try classification
            # scores = timbre_classification(instruments, inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test, display=1, class_label=class_label, display_label = display_label)
            # scores = timbre_classification(instruments, inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test, display=1)
            scores, clf = timbre_classification(instruments, inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test, display=1, return_model=1)
            # pltX = np.array(nH_list)
            # pltXLabel = 'Harmonics number'
            # trainAccuracy.append(scores[0])
            # testAccuracy.append(scores[1])

            # try PCA
            # for n_components in decomp_list:
            #     scores = feature_PCA(instruments, inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test, n_components=n_components, display=0)
            #     trainAccuracy.append(scores[0])
            #     testAccuracy.append(scores[1])
            # pltX = np.array(decomp_list)
            # pltXLabel = 'principal component number'

            # try LDA
            # scores = feature_LDA(instruments, inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test)
           
            # save the classifier
            save_model(clf, 'model/rf_8ins_40harm.joblib')

            # test clf loading
            clf_load = load('model/rf_8ins_40harm.joblib')
            testScore = clf_load.score(inputTest.T, labelTest[0,:])
            print(testScore)
            
            # plot error
            # harm = np.arange(allErrTrain.shape[1])
            # plt.plot(harm,allErrTrain[0,:],'k-')

            # UM.save_matrix('../temp.xlsx', allErrTrain.T, sheetName='error_train')
            # UM.save_matrix('../temp.xlsx', allErrTest.T, sheetName='error_test')
            # print(allErr)

        # plotting
        # plt.plot(pltX,np.array(trainAccuracy),pltX,np.array(testAccuracy))
        # plt.plot(pltX,np.array(testAccuracy),'k*')
        # plt.xlabel(pltXLabel)
        # plt.ylabel('Accuracy')
        # plt.legend(['Train','Test'])
        # plt.show()
        
        # test_dir = 'output_sounds/'
        # test_fts = []
        # sd_names = []
        # index = 0
        # for f in os.listdir(test_dir):
        #     if f[-4:] == '.wav':            

        #         pitch = 69

        #         M = 4096
        #         N = M*2
        #         Ns = 512
        #         H = 256
        #         nH = 40
        #         sdInfo = {
        #             'instrument':'synthesis',
        #             'pitch':str(pitch),
        #             'source':'' ,
        #             'index':index,
        #             'nH':nH,
        #             'nF': 0,
        #             'FFTLenAna': N,
        #             'FFTLenSyn': Ns,
        #             'hopSize': H
        #         }
        #         index = index+1

        #         freqency = UM.pitch2freq(pitch)
        #         minf0 = freqency-50
        #         maxf0 = minf0+100

        #         print(test_dir+f)

        #         sdInfo = TM.sound_info_clct(test_dir+f, sdInfo, nH=nH, minf0=minf0, maxf0=maxf0, M=M, N=N, Ns=Ns, H=H)    
        #         # UM.save_dictionary(sdInfo,'../../sounds/W13sd/fts/', fm=1)
        #         # f_comp = os.path.join(test_dir, f)
        #         # print(index)

        #         ft = json2vector(sdInfo,nH=nH)
        #         test_fts.append(ft)
        #         sd_names.append(f)
        # test_fts = np.array(test_fts)
        # test_y = clf.predict(test_fts[:,1:])

        # for i in range(len(sd_names)):
        #     print(sd_names[i],instruments[test_y[i]])
        


    # listen to single feature modification
    elif exp == 4:
        # file_path = '../../sounds/flute_acoustic/flute_acoustic_valid/flute_acoustic_002-084-050.wav'
        # file_path = '../../sounds/brass_acoustic/brass_acoustic_valid/brass_acoustic_006-069-075.wav'
        # file_path = '../../sounds/string_acoustic/string_acoustic_valid/string_acoustic_012-048-050.wav'
        # file_path = '../../sounds/reed_acoustic/reed_acoustic_valid/reed_acoustic_037-069-050.wav'

        # file_path = '../../sounds/phiharmonia/violin/violin_A5_15_forte_arco-normal.wav'
        # file_path = '../../sounds/phiharmonia/flute/flute_A5_15_forte_normal.wav'
        # file_path = '../../sounds/phiharmonia/trumpet/trumpet_A4_15_forte_normal.wav'
        # file_path = '../../sounds/phiharmonia/clarinet/clarinet_A4_15_forte_normal.wav'
        # file_path = '../../sounds/phiharmonia/cello/cello_A2_15_forte_arco-normal.wav'

        # UM.plot_spectrogram(file_path)

        # sd_path = 'result/features/flute_acoustic/flute_acoustic_valid/flute_acoustic-C6-002-050.json'
        # sd_path = 'result/features/brass_acoustic/brass_acoustic_valid/brass_acoustic-A4-006-075.json'
        # sd_path = 'result/features/string_acoustic/string_acoustic_valid/string_acoustic-C3-012-050.json'
        # sd_path = 'result/features/reed_acoustic/reed_acoustic_valid/reed_acoustic-A4-037-050.json'
        # sd_path = 'result/features/brass_acoustic/brass_acoustic_valid/brass-A4-0.json'

        # sd_path = 'result/features/phiharmonia/violin/violin-81-4.json'
        # sd_path = 'result/features/phiharmonia/flute/flute-81-4.json' 
        # sd_path = 'result/features/phiharmonia/trumpet/trumpet-69-4.json'
        # sd_path = 'result/features/phiharmonia/clarinet/clarinet-69-4.json'
        # sd_path = 'result/features/phiharmonia/cello/cello-45-4.json'

        ft = json2vector(sd_path,nH=40)
        sdInfo_original = UM.read_features(sd_path,fm=1)
        sdInfo = vector2dict([ft[0]], ft[1:], nF=sdInfo_original['nF'], silenceSt=10, silenceEd=10, meanMax=-40, nH=40)

        # assign original stochastic values to sdInfo
        sdInfo['stocEnv'] = sdInfo_original['stocEnv']

        # check sdInfo
        for key in sdInfo:
            print(key)
            print('oringinal:', sdInfo_original[key])
            print('recoverred:', sdInfo[key])
            

        y, yh, yst, hfreqSyn, hmagSyn, hphaseSyn = TM.sound_syn_from_para(sdInfo)
        y_json, yh_json, yst_json, hfreqSyn_json, hmagSyn_json, hphaseSyn_json = TM.sound_syn_from_para(sdInfo_original)

        UF.wavplay(file_path)
        UM.display_syn(file_path, sdInfo_original['fs'],sdInfo_original['hopSize'], sdInfo['nF'], y_json, yh_json, yst_json, hfreqSyn_json, hmagSyn_json, hphaseSyn_json, mode=1)
        UM.display_syn(file_path, sdInfo['fs'], sdInfo['hopSize'], sdInfo['nF'], y, yh, yst, hfreqSyn, hmagSyn, hphaseSyn,mode=1)
        # UM.plot_spectrogram(y,sdInfo['fs'])


    #################################################################################################

    elif exp == 5:
        # instruments = ['flute','string','brass','reed']
        # instruments = ['flute','oboe','clarinet','french horn','trumpet','violin','cello']
        instruments = ['flute', 'violin','trumpet','clarinet']

        # create a inital kernel for k_means
        init_pathes = ['result/features/phiharmonia/flute/flute-69-4.json',
                'result/features/phiharmonia/violin/violin-70-4.json',
                'result/features/phiharmonia/trumpet/trumpet-69-4.json',
                'result/features/phiharmonia/clarinet/clarinet-69-4.json']
        
        nH = 40
        maxFreq = 551
        init_ft = []

        # set initial kernels
        if len(instruments) == len(init_pathes):
            for f in init_pathes:
                ft = json2vector(f,0,nH)
                init_ft.append(ft[1:])
            init_ft = np.array(init_ft)
        else:
            print('classes number and initial kernels number not match')

        # collect features
        inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test = get_mixed_ins_data(instruments,nH=nH, source=1,maxFreq = maxFreq) 
        centroid, label, inertia = k_means(np.hstack((inputTrain,inputTest)).T, n_clusters=len(instruments), init=init_ft, max_iter=1000)
        
        # compare the two
        conf_mat = confusion_matrix(np.concatenate((labelTrain[0,:],labelTest[0,:])), label)
        print('confusion matrix, random initial kernel:\n',conf_mat)


    elif exp == 6: 
        # morphing

        # file_path1 = '../../sounds/phiharmonia/violin/violin_A5_15_forte_arco-normal.wav'
        # file_path2 = '../../sounds/phiharmonia/flute/flute_A5_15_forte_normal.wav'
        # sd_path1 = 'result/features/phiharmonia/violin/violin-81-4.json'
        # sd_path2 = 'result/features/phiharmonia/flute/flute-81-4.json'

        file_path1 = '../../sounds/phiharmonia/trumpet/trumpet_A4_15_forte_normal.wav'
        file_path2 = '../../sounds/phiharmonia/clarinet/clarinet_A4_15_forte_normal.wav'
        sd_path1 = 'result/features/phiharmonia/trumpet/trumpet-69-4.json'
        sd_path2 = 'result/features/phiharmonia/clarinet/clarinet-69-4.json'
        
        sdInfo1 = UM.read_features(sd_path1,fm=1)
        sdInfo2 = UM.read_features(sd_path2,fm=1)
        sdInfo = UM.read_features(sd_path1,fm=1)

        morph_rate = 0.1

        for key in sdInfo1:
            if key not in ['instrument', 'pitch', 'source', 'index', 'nH', 'FFTLenAna', 'FFTLenSyn', 'hopSize', 'fs', 'stocEnv']:
                sdInfo[key] = morph_rate*sdInfo1[key] + (1-morph_rate)*sdInfo2[key]
                if key in ['nF','freqSmoothLen']:
                    sdInfo[key] = int(sdInfo[key])
                elif key in ['magADSRIndex']:
                    sdInfo[key] = np.array(sdInfo[key],dtype=int)
        sdInfo['nF'] = int(sdInfo['nF'])


        y, yh, yst, hfreqSyn, hmagSyn, hphaseSyn = TM.sound_syn_from_para(sdInfo)

        # UF.wavplay(file_path1)
        # UF.wavplay(file_path2)
        UM.display_syn(file_path1, sdInfo['fs'],sdInfo['hopSize'], sdInfo['nF'], y, yh, yst, hfreqSyn, hmagSyn, hphaseSyn, mode=1)
  


