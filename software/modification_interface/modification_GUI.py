'''
modification main window
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../modification/'))

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi
import numpy as np

import sound_info as SI
import drag_point_GUI as DP
import utilModi as UM

class modification_GUI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.initUI()
    
    def initUI(self):
        # initiate the window
        loadUi('modification_GUI.ui',self)
        self.setWindowTitle('Modification GUI')

        # initiate sound information section
        self.init_sound_information()

        # initiate sound features section
        self.init_sound_features()

        # initiate sound morphing section
        self.init_sound_morphing()


    def init_sound_information(self):
        # initiate sound path
        self.sound_path_text.setPlainText('../../sounds/A4/trumpet-A4.wav')
        self.file_path = '../../sounds/A4/trumpet-A4.wav'
        self.sound_path_button.clicked.connect(self.click_file_path_button)

        # initiate combo box for the window type
        self.window_type_box.addItems(['rectangular','triangle','hanning','hamming','blackman','blackmanharris'])
        self.window_type = 'boxcar'
        self.window_type_box.currentIndexChanged.connect(self.itemChanged_window_type)

        # initiate window size text
        self.window_size_text.setPlainText('4096')

        # initiate fft size text
        self.fft_size_text.setPlainText('4096')
        
        # initiate hop size text
        self.hop_size_text.setPlainText('256')

        # initiate fundamental frequency or pitch text
        self.fundamental_frequency_text.setPlainText('A4')

        # initiate maximum harmonics number text
        self.max_harmonics_number_text.setPlainText('40')

        # initiate compute button
        self.compute_button.clicked.connect(self.click_compute_button)

        # initiate display button
        self.display_button.clicked.connect(self.click_display_button)

        # initiate a sound_info instance
        self.update_sound_info()

    def init_sound_features(self):
        # initiate harmonics number selection
        self.harmNo_box.setMinimum(0)
        self.harmNo_box.setMaximum(self.nH)
        self.harmNo = 0 # default value
        # self.show_features()
        self.harmNo_box.valueChanged.connect(self.valueChanged_harmNo)

        # initiate drag point button
        self.drag_points_button.clicked.connect(self.click_drag_points_button)

        # initiate update button
        self.update_button.clicked.connect(self.click_update_button)

        # initiate play button
        self.play_button.clicked.connect(self.click_play_button)

    def init_sound_morphing(self):
        # initiate reference combo box 
        self.reference_box.addItems(['flute','oboe','clarinet','saxophone','french horn','trumpet','violin','cello'])

        # initiate morphing rate
        self.morphing_rate_text.setPlainText('0.50')
        self.morphing_rate_text.textChanged.connect(self.textChanged_morphing_rate_text)
        self.morphing_rate_slider.valueChanged.connect(self.valueChanged_morphing_rate_slider)

        # set duration text
        self.duration_text.setPlainText('1')

        # set classification text
        # self.classification_text.setPlainText('french horn')

        # initiate morph button
        self.morph_button.clicked.connect(self.click_morph_button)
        
        # initiate play morph button
        self.play_morph_button.clicked.connect(self.click_play_morph_button)

        
        
    def click_file_path_button(self):
        # slot function for clicking the browse button in the file path options
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.AnyFile)
        fd.setFilter(QDir.Files)
        
        if fd.exec_():
            fp = fd.selectedFiles()[0]
            if fp[-4:] != '.wav':
                msg = QMessageBox()
                msg.setWindowTitle('Message Box')
                msg.setText('This is not a .wav file! Please choose another file.')

            else:
                self.file_path = fp
                self.sound_path_text.setText(fp)

    def itemChanged_window_type(self):
        # slot function for item changing of the window type combo box
        window_type = ['boxcar','triang','hann','hamming','blackman','blackmanharris']
        self.window_type = window_type[self.window_type_box.currentIndex()]

    def click_compute_button(self):
        # slot function for clicking the compute button
        # get the contents of the EditTexts
        self.update_sound_info()        
        self.draw_widgets()
        self.show_features()

    def draw_widgets(self):
        # show the spectrogram and harmonics plotting
        # orginal spectrogram
        max_plot_freq = 5000.0
        mX_original = self.sound_information.get_spectrogram(source='original')
        numFrames = int(mX_original[:,0].size)
        fs = float(self.sound_information.sdInfo['fs'])
        frmTime = self.H * np.arange(numFrames) / fs 
        binFreq = fs * np.arange(self.N * max_plot_freq / fs) / self.N
        self.spectrogram_original_display_widget.canvas.axes.clear()
        self.spectrogram_original_display_widget.canvas.axes.pcolormesh(frmTime, binFreq, np.transpose(mX_original[:, :int(self.N * max_plot_freq / fs + 1)]))
        self.spectrogram_original_display_widget.canvas.axes.set_xlabel('time (sec)')
        self.spectrogram_original_display_widget.canvas.axes.set_ylabel('frequency (Hz)')
        self.spectrogram_original_display_widget.canvas.axes.set_title('original magnitude spectrogram')
        self.spectrogram_original_display_widget.canvas.axes.set_autoscale_on(True)
        self.spectrogram_original_display_widget.canvas.draw()

        # synth spectrogram
        mX_synth = self.sound_information.get_spectrogram(source='synth')
        numFrames = int(mX_synth[:,0].size)
        frmTime = self.sound_information.sdInfo['hopSize'] * np.arange(numFrames) / fs
        binFreq = fs * np.arange(self.N * max_plot_freq / fs) / self.N
        self.spectrogram_synth_display_widget.canvas.axes.clear()
        self.spectrogram_synth_display_widget.canvas.axes.pcolormesh(frmTime, binFreq, np.transpose(mX_synth[:, :int(self.N * max_plot_freq / fs + 1)]))
        self.spectrogram_synth_display_widget.canvas.axes.set_xlabel('time (sec)')
        self.spectrogram_synth_display_widget.canvas.axes.set_ylabel('frequency (Hz)')
        self.spectrogram_synth_display_widget.canvas.axes.set_title('synthetic magnitude spectrogram')
        self.spectrogram_synth_display_widget.canvas.axes.set_autoscale_on(True)
        self.spectrogram_synth_display_widget.canvas.draw()

        # original harmonics 3d plotting
        max_plot_harmNo = min(self.sound_information.hfreq.shape[1],10)
        t = np.arange(self.sound_information.sdInfo_org['nF']) * self.H / fs
        self.harmonics_original_display_widget.canvas.axes.clear()
        for i in range(max_plot_harmNo):
            self.harmonics_original_display_widget.canvas.axes.plot(t, self.sound_information.hfreq[::,i], self.sound_information.hmag[::,i]) 
        self.harmonics_original_display_widget.canvas.axes.set_xlabel('time (sec)')
        self.harmonics_original_display_widget.canvas.axes.set_ylabel('frequency (Hz)')
        self.harmonics_original_display_widget.canvas.axes.set_zlabel('magnitude (dB)')
        self.harmonics_original_display_widget.canvas.axes.set_title('original harmonics')
        self.harmonics_original_display_widget.canvas.axes.set_autoscale_on(True)
        self.harmonics_original_display_widget.canvas.draw()

        # synthetic harmonics 3d plotting
        self.harmonics_synth_display_widget.canvas.axes.clear()
        t = np.arange(self.sound_information.sdInfo['nF']) * self.H / fs
        for j in range(max_plot_harmNo):
            self.harmonics_synth_display_widget.canvas.axes.plot(t, self.sound_information.hfreqSyn[::,j], self.sound_information.hmagSyn[::,j]) 
        self.harmonics_synth_display_widget.canvas.axes.set_xlabel('time (sec)')
        self.harmonics_synth_display_widget.canvas.axes.set_ylabel('frequency (Hz)')
        self.harmonics_synth_display_widget.canvas.axes.set_zlabel('magnitude (dB)')
        self.harmonics_synth_display_widget.canvas.axes.set_title('synthetic harmonics')
        self.harmonics_synth_display_widget.canvas.axes.set_autoscale_on(True)
        self.harmonics_synth_display_widget.canvas.draw()

    def update_sound_info(self):
        self.M = int(self.window_size_text.toPlainText())
        self.N = int(self.fft_size_text.toPlainText())
        self.H = int(self.hop_size_text.toPlainText())
        self.pitch = self.fundamental_frequency_text.toPlainText()
        try:
            self.pitch = int(self.pitch)
        except:
            pass
        self.nH = int(self.max_harmonics_number_text.toPlainText())

        if not hasattr(self, 'sound_information'):
            self.sound_information = SI.sound_info(self.file_path, self.pitch, self.N, self.M, self.H, self.nH, self.window_type)
        else:
            self.sound_information.update(self.file_path, self.pitch, self.N, self.M, self.H, self.nH, self.window_type)
    
    def click_display_button(self):
        # slot functinon for clicking the display button
        self.sound_information.display_original_sound()
        
    def valueChanged_harmNo(self):
        self.harmNo = self.harmNo_box.value()

        # check if the harmonic can be detected
        freq_mean = self.sound_information.sdInfo['freqMean'][self.harmNo]
        if freq_mean != freq_mean: # means freq_mean is NaN
            msg = QMessageBox()
            msg.setWindowTitle('Message Box')
            msg.setText('This harmonic cannot be detected since the frequency is out of the range of Nyquist frequency. Plese choose a smaller number.')

        else:
            # show the sound features in the text edit boxes
            self.show_features()

    def show_features(self):
        # frequency
        self.freq_mean_text.setPlainText(str(round(self.sound_information.sdInfo['freqMean'][self.harmNo],2)))
        self.freq_var_text.setPlainText(str(round(self.sound_information.sdInfo['freqVar'][self.harmNo],2)))

        # magnitude time
        key_point_time = np.round(self.sound_information.sdInfo['magADSRIndex'][:,self.harmNo] * self.H / float(self.sound_information.sdInfo['fs']),2)
        self.SOA_time_text.setPlainText(str(key_point_time[0]))
        self.EOA_time_text.setPlainText(str(key_point_time[1]))
        self.SOR_time_text.setPlainText(str(key_point_time[2]))
        self.EOR_time_text.setPlainText(str(key_point_time[3]))

        # magnitude amplitude
        key_point_amp = np.round(self.sound_information.sdInfo['magADSRValue'][:,self.harmNo],2)
        self.SOA_amp_text.setPlainText(str(key_point_amp[0]))
        self.EOA_amp_text.setPlainText(str(key_point_amp[1]))
        self.SOR_amp_text.setPlainText(str(key_point_amp[2]))
        self.EOR_amp_text.setPlainText(str(key_point_amp[3]))

        # magnitude segment shape
        seg_n = np.round(self.sound_information.sdInfo['magADSRN'][:,self.harmNo],2)
        self.start_shape_text.setPlainText(str(seg_n[0]))
        self.attack_shape_text.setPlainText(str(seg_n[1]))
        self.steady_shape_text.setPlainText(str(seg_n[2]))
        self.release_shape_text.setPlainText(str(seg_n[3]))
        self.end_shape_text.setPlainText(str(seg_n[4]))

        # first frame phase
        self.phase_slope_text.setPlainText(str(round(self.sound_information.sdInfo['phaseffSlope'],2)))
        self.phase_intercept_text.setPlainText(str(round(self.sound_information.sdInfo['phaseffIntercept'],2)))

    def click_drag_points_button(self):        
        self.dialog = DP.drag_point_GUI(sound_information=self.sound_information, harmNo=self.harmNo)
        self.dialog.signal.connect(self.get_drag_points_data)
        self.dialog.exec_()

    def get_drag_points_data(self, xInd, x, y):
        # update the key points' indexs in sdInfo
        # print('get data!')
        self.sound_information.sdInfo['magADSRIndex'][:,self.harmNo] = xInd

        # update the key points' time to the text boxes
        self.SOA_time_text.setPlainText(str(round(x[0],2)))
        self.EOA_time_text.setPlainText(str(round(x[1],2)))
        self.SOR_time_text.setPlainText(str(round(x[2],2)))
        self.EOR_time_text.setPlainText(str(round(x[3],2)))

        # update the key points' values
        self.sound_information.sdInfo['magADSRValue'][:,self.harmNo] = y
        self.SOA_amp_text.setPlainText(str(round(y[0],2)))
        self.EOA_amp_text.setPlainText(str(round(y[1],2)))
        self.SOR_amp_text.setPlainText(str(round(y[2],2)))
        self.EOR_amp_text.setPlainText(str(round(y[3],2)))

        # update the plots
        self.sound_information.get_synInfo()
        self.draw_widgets()

        
        

    def click_update_button(self):
        # slot function for clicking the update button
        self.harmNo = self.harmNo_box.value()
        
        # frequency
        self.sound_information.sdInfo['freqMean'][self.harmNo] = float(self.freq_mean_text.toPlainText())
        self.sound_information.sdInfo['freqVar'][self.harmNo] = float(self.freq_var_text.toPlainText())

        # magnitude time
        key_point_time = []
        key_point_time.append(int(float(self.SOA_time_text.toPlainText())/ self.H * float(self.sound_information.sdInfo['fs'])))
        key_point_time.append(int(float(self.EOA_time_text.toPlainText())/ self.H * float(self.sound_information.sdInfo['fs'])))
        key_point_time.append(int(float(self.SOR_time_text.toPlainText())/ self.H * float(self.sound_information.sdInfo['fs'])))
        key_point_time.append(int(float(self.EOR_time_text.toPlainText())/ self.H * float(self.sound_information.sdInfo['fs'])))
        self.sound_information.sdInfo['magADSRIndex'][:,self.harmNo] = key_point_time

        # magnitude amplitude
        key_point_amp = []
        key_point_amp.append(float(self.SOA_amp_text.toPlainText()))
        key_point_amp.append(float(self.EOA_amp_text.toPlainText()))
        key_point_amp.append(float(self.SOR_amp_text.toPlainText()))
        key_point_amp.append(float(self.EOR_amp_text.toPlainText()))
        self.sound_information.sdInfo['magADSRValue'][:,self.harmNo] = key_point_amp

        # magnitude segment shape
        seg_n = []
        seg_n.append(float(self.start_shape_text.toPlainText()))
        seg_n.append(float(self.attack_shape_text.toPlainText()))
        seg_n.append(float(self.steady_shape_text.toPlainText()))
        seg_n.append(float(self.release_shape_text.toPlainText()))
        seg_n.append(float(self.end_shape_text.toPlainText()))
        self.sound_information.sdInfo['magADSRN'][:,self.harmNo] = seg_n

        # phase
        self.sound_information.sdInfo['phaseffSlope'] = float(self.phase_slope_text.toPlainText())
        self.sound_information.sdInfo['phaseffIntercept'] = float(self.phase_intercept_text.toPlainText())

        # synthesize sound
        self.sound_information.get_synInfo()

        # update the plotting
        self.draw_widgets()

    def click_play_button(self):
        # slot function of clicking the play button
        self.sound_information.display_synth_sound()

    
    def valueChanged_morphing_rate_slider(self):
        self.morphing_rate_text.setPlainText(str(round(self.morphing_rate_slider.value()/100,2)))

    def textChanged_morphing_rate_text(self):
        self.morphing_rate_slider.setValue(int(float(self.morphing_rate_text.toPlainText())*100))

    def click_morph_button(self):

        # get sound imformation of the reference sound
        self.reference = self.reference_box.currentText()
        if self.reference == 'french horn':
            self.reference = 'french-horn'
        if self.get_ref_sdInfo(): # reference sound exists
            

            # get morphing rate
            self.morph_rate = float(self.morphing_rate_text.toPlainText())

            # get duration
            self.duration = int(float(self.duration_text.toPlainText()) / self.sound_information.sdInfo['hopSize'] * self.sound_information.sdInfo['fs'])

            # get intensity (ranging from ~-150dB to ~-30dB)
            self.intensity = self.intensity_slider.value()
            
            # get the morphed sound information
            self.sound_information.get_sdInfo_morph(self.sdInfo_ref, self.morph_rate, self.duration, self.intensity)

            # update sound 
            self.sound_information.get_synInfo()
            self.draw_widgets()

            # do classification
            self.sound_information.get_class()
            self.classification_text.setPlainText(self.sound_information.class_res)        

    def get_ref_sdInfo(self):
        files_path = 'sounds_features/'
        
        # get the file name
        if isinstance(self.pitch, str):
            file_name = self.reference + '-' + str(UM.pitchname2num(self.pitch,'#')) + '.json'
        else:
            file_name = self.reference + '-' + str(UM.freq2pitch(self.pitch)) + '.json'

        # read the json file
        if os.path.exists(files_path + file_name):
            self.sdInfo_ref = UM.read_features(files_path+file_name, fm=1)
            return 1
        else: 
            msg = QMessageBox()
            msg.setWindowTitle('Message Box')
            msg.setText('Oh No! Cannot find the reference sound...')
            return 0

    def click_play_morph_button(self):
        self.sound_information.display_synth_sound()


if __name__ == '__main__':
    app = QApplication([])
    window = modification_GUI()
    window.show()
    app.exec_()

