'''  
modification drag point QDialog 
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../modification/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../models/'))
sys.path.append('/Library/Python/2.7/site-packages/')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from matplotlib.lines import Line2D
from matplotlib.artist import Artist

import utilModi as UM
import sound_info as SI

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi

class drag_point_GUI(QDialog):
    # define a signal
    signal = pyqtSignal(list, list, list)

    def __init__(self,parent=None,sound_information=None,harmNo=None):
        super(drag_point_GUI, self).__init__(parent)
        loadUi('drag_point_GUI.ui', self)
        self.setWindowTitle('Drag harmonic key points')

        # initiate the drag point QDialog
        self.set_parameters(sound_information, harmNo)
        self.initUI()

        # self.exec_()

    def set_parameters(self, sound_information, harmNo):
        self.sound_information = sound_information
        self.harmNo = harmNo

    def initUI(self):
        # initiate the drag point widget
        self.drag_widget.init_widget(self.sound_information, self.harmNo)

        # connect the signal and the slot
        self.buttonBox.accepted.connect(self.slot_return_xy)

    def slot_return_xy(self):
        self.signal.emit(self.drag_widget.xInd.tolist(), self.drag_widget.x.tolist(), self.drag_widget.y.tolist())
        
        # tst
        # print('slot triggered')
