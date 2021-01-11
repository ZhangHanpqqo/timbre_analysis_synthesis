import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../transformations/'))
sys.path.append('/Library/Python/2.7/site-packages/')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error,classification_report,plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm

import utilModi as UM
import utilFunctions as UF
import timbreModi as TM

def json2vector(file_path,raw=0,nH=40):
    if os.path.exists(file_path):
        sdInfo = UM.read_features(file_path,fm=1)
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
                     + np.reshape(sdInfo['magADSRN'][:,:harmClct],-1).tolist() + [sdInfo['phaseffSlope'],sdInfo['phaseffIntercept']]

            elif raw == 1:
                magADSRValue = dB2amp(sdInfo['magADSRValue'])
                ft = [sdInfo['f0']] + sdInfo['freqMean'][:harmClct].tolist() + sdInfo['freqVar'][:harmClct].tolist() + attackTimeAbs[:harmClct].tolist() \
                     + releaseTimeRlt[:harmClct].tolist() + magADSRValue.tolist() \
                     + np.reshape(sdInfo['magADSRN'][:, :harmClct], -1).tolist() + [sdInfo['phaseffSlope'],sdInfo['phaseffIntercept']]

            else:
                print('wrong raw parameter, returning raw date.')
                magADSRValue = dB2amp(sdInfo['magADSRValue'])
                ft = [sdInfo['f0']] + sdInfo['freqMean'][:harmClct].tolist() + sdInfo['freqVar'][:harmClct].tolist() + attackTimeAbs[:harmClct].tolist() \
                     + releaseTimeRlt[:harmClct].tolist() + magADSRValue.tolist() \
                     + np.reshape(sdInfo['magADSRN'][:, :harmClct], -1).tolist() + [sdInfo['phaseffSlope'],sdInfo['phaseffIntercept']]
            return ft
        else:
            print("Need"+ str(harmClct)+ "harmonics, only ",sdInfo['nH']," collected!")
            return

    else:
        print("File not found!")
        return

def get_feature_list(dir,fts,raw,nH):
    newDir = dir
    if os.path.isfile(dir) and dir[-4:] == 'json':
        ft = json2vector(dir,raw,nH)
        fts.append(ft)
    elif os.path.isdir(dir):
        for f in os.listdir(dir):
            newDir = os.path.join(dir,f)
            fts = get_feature_list(newDir,fts,raw,nH)
    return fts

def feature_vectorization(files_path,raw = 0,nH = 40):
    fts = []
    fts = get_feature_list(files_path,fts,raw,nH)

    return np.array(fts).T

def vector2dict(x,Y,nF=1000,silenceSt = 10, silenceEd = 10, meanMax = -100,nH = 40):
    sdInfo = {'instrument':'synthesis',
              'pitch':'',
              'source':'',
              'index':'111',
              'nH':nH,
              'nF':nF,
              'FFTLenAna':8192,
              'FFTLenSyn':512,
              'hopSize':128,
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
    magADSRN = np.reshape(Y[cursor:cursor+5*nH],(5,-1))
    cursor += 5*nH
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
def get_mixed_ins_data(instruments, nH=40):
    for ins in instruments:
        train_path = 'result/features/' + ins + '/' + ins + '_valid'
        test_path = 'result/features/' + ins + '/' + ins + '_test'
        ftMatTrain = feature_vectorization(train_path, nH=nH)
        ftMatTest = feature_vectorization(test_path, nH=nH)
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

        print(ins,' data collecting finished.')

    return inputTrain,labelTrain.astype(int),f0Train,inputTest,labelTest.astype(int),f0Test

def timbre_classification(instruments, inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test):
    # clf = svm.SVC(decision_function_shape='ovr',probability=False)
    clf = RandomForestClassifier(n_estimators=80,random_state=0)
    clf.fit(inputTrain.T,labelTrain[0,:])
    YTrain = clf.predict(inputTrain.T).astype(int)
    YTest = clf.predict(inputTest.T).astype(int)

    print("Train",clf.score(inputTrain.T,labelTrain[0,:]))
    print("Test", clf.score(inputTest.T, labelTest[0, :]))

    # print(classification_report(labelTrain[0,:],YTrain,instruments))
    # print(classification_report(labelTest[0, :],YTest, instruments))
    # disp = plot_confusion_matrix(clf, inputTrain.T, labelTrain[0, :])
    # plt.show()
    # print("Train Confusion matrix:\n%s" % disp.confusion_matrix)
    disp = plot_confusion_matrix(clf,inputTest.T,labelTest[0, :])
    # print("Test Confusion matrix:\n%s" % disp.confusion_matrix)
    plt.show()

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

def feature_PCA(instruments, inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test):
    n_ft, n_num_train = inputTrain.shape
    _,n_num_test = inputTest.shape

    # normalization (train+test)
    normMax = np.array([np.max(np.abs(inputTrain),axis=1),np.max(np.abs(inputTest),axis=1)])
    inputMaxRep = np.repeat(np.array([np.max(normMax,axis=0)]).T,n_num_train,axis=1)
    inputTrainNorm = inputTrain/inputMaxRep
    inputMaxRep = np.repeat(np.array([np.max(normMax, axis=0)]).T, n_num_test, axis=1)
    inputTestNorm = inputTest/inputMaxRep

    pca = PCA(n_components=3)
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

    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(corr),vmin=0,vmax=1)
    clb = ax.figure.colorbar(im, ax=ax)
    clb.ax.set_ylabel("", rotation=-90, va="bottom")
    plt.show()

    # do classification
    timbre_classification(instruments, inputTrainDecomp.T, labelTrain, f0Train, inputTestDecomp.T, labelTest, f0Test)

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
    print('Train ',clf.score(inputTrainNorm.T,labelTrain[0,:]))
    print('Test ', clf.score(inputTestNorm.T, labelTest[0, :]))

    # print confusion matrix
    disp = plot_confusion_matrix(clf, inputTrainNorm.T, labelTrain[0, :])
    plt.show()
    # print("Train Confusion matrix:\n%s" % disp.confusion_matrix)
    disp = plot_confusion_matrix(clf, inputTestNorm.T, labelTest[0, :])
    plt.show()


if __name__ == '__main__':

    # 1: regression for all features for one instrument
    # 2: max magnitude key point clustering for one instrument
    # 3: classification for different instruments
    exp = 4


    if exp == 1: # regression for all features for one instrument
        train_path = 'result/features/brass_acoustic/brass_acoustic_valid'
        ftMatTrain = feature_vectorization(train_path)
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
        instruments = ['flute','string','brass','reed']
        # instruments = ['brass', 'reed']

        # get inputs and labels
        for i in range(len(instruments)):
            instruments[i] = instruments[i]+'_acoustic'
        inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test = get_mixed_ins_data(instruments,nH=6)

        # select features based on timbre space: rise time, and mag values
        # inputTrain = np.vstack((inputTrain[12:18,:],inputTrain[24:48,:]))
        # inputTest = np.vstack((inputTest[12:18,:],inputTest[24:48,:]))

        # select the complementary of timbre space
        # inputTrain = np.vstack((inputTrain[:12,:],inputTrain[18:24,:],inputTrain[48:,:]))
        # inputTest = np.vstack((inputTest[:12, :], inputTest[18:24, :], inputTest[48:, :]))

        # try classification
        timbre_classification(instruments, inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test)

        # try PCA
        # feature_PCA(instruments, inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test)

        # try LDA
        # feature_LDA(instruments, inputTrain, labelTrain, f0Train, inputTest, labelTest, f0Test)

    # plot error
    # harm = np.arange(allErrTrain.shape[1])
    # plt.plot(harm,allErrTrain[0,:],'k-')

    # UM.save_matrix('../temp.xlsx', allErrTrain.T, sheetName='error_train')
    # UM.save_matrix('../temp.xlsx', allErrTest.T, sheetName='error_test')
    # print(allErr)

    # listen to single feature modification
    elif exp == 4:
        # file_path = '../../sounds/flute_acoustic/flute_acoustic_valid/flute_acoustic_002-084-050.wav'
        file_path = '../../sounds/brass_acoustic/brass_acoustic_valid/brass_acoustic_006-069-075.wav'
        # file_path = '../../sounds/string_acoustic/string_acoustic_valid/string_acoustic_012-048-050.wav'
        # file_path = '../../sounds/reed_acoustic/reed_acoustic_valid/reed_acoustic_037-069-050.wav'
        # UM.plot_spectrogram(file_path)

        # sd_path = 'result/features/flute_acoustic/flute_acoustic_valid/flute_acoustic-C6-002-050.json'
        # sd_path = 'result/features/brass_acoustic/brass_acoustic_valid/brass_acoustic-A4-006-075.json'
        # sd_path = 'result/features/string_acoustic/string_acoustic_valid/string_acoustic-C3-012-050.json'
        # sd_path = 'result/features/reed_acoustic/reed_acoustic_valid/reed_acoustic-A4-037-050.json'
        sd_path = 'result/features/brass_acoustic/brass_acoustic_valid/brass-A4-0.json'

        ft = json2vector(sd_path,nH=10)
        sdInfo = vector2dict([ft[0]], ft[1:], nF=1000, silenceSt=10, silenceEd=10, meanMax=-50, nH=10)

        # check sdInfo
        for key in sdInfo:
            print(key, sdInfo[key])

        y, yh, yst, hfreqSyn, hmagSyn, hphaseSyn = TM.sound_syn_from_para(sdInfo)
        UM.display_syn(file_path, sdInfo['fs'], sdInfo['hopSize'], sdInfo['nF'], y, yh, yst, hfreqSyn, hmagSyn, hphaseSyn,mode=3)
        UM.plot_spectrogram(y,sdInfo['fs'])


    #################################################################################################

    ################## test function vector2dict() ##################
    # file_path = 'result/features/flute_acoustic/flute_acoustic-A#6-002.json'
    # ft = json2vector(file_path)
    # # print(ft[281:321])
    # sdInfo = vector2dict([ft[0]],ft[1:],nF=1000,silenceSt = 10, silenceEd = 10, meanMax = -100)
    # for i in sdInfo:
    #     print(i,sdInfo[i])
    #
    # y, yh, yst, hfreqSyn, hmagSyn, hphaseSyn = TM.sound_syn_from_para(sdInfo)
    #
    # UM.display_syn('../../sounds/flute_acoustic/flute_acoustic_002-094-127.wav', sdInfo['fs'], sdInfo['hopSize'], sdInfo['nF'], y, yh, yst, hfreqSyn, hmagSyn, hphaseSyn,
    #                mode=1)
    #################################################################



