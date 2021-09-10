'''  Utility classes and functions for modification GUI   
     Class: point_drag_axes, point_drag_fig               '''
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

from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

class drag_point_widget(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.canvas = FigureCanvas(Figure())

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)

        self.canvas.ax = self.canvas.figure.add_subplot(111)
        # self.init_widget(self)
        self.setLayout(vertical_layout)

    def init_widget(self, sound_information, harmNo):
        # set some parameters
        self.showverts = True
        self.offset = 5
        self.sound_information = sound_information
        self.harmNo = harmNo
        self.t = np.arange(self.sound_information.sdInfo['nF']) * self.sound_information.sdInfo['hopSize'] / self.sound_information.sdInfo['fs']

        # set title
        self.canvas.ax.set_title('Harmonic #'+str(self.harmNo)+' time-magnitude')
        # set limitations
        self.canvas.ax.set_xlim((self.t[0]-0.1, self.t[-1]+0.1))
        self.canvas.ax.set_ylim((-155, 5))
        # set labels
        self.canvas.ax.set_xlabel('time(s)')
        self.canvas.ax.set_ylabel('magnitude')

        # set initial parameters
        self.harm = self.sound_information.hmagSyn[:,self.harmNo]
        self.xInd = self.sound_information.sdInfo['magADSRIndex'][:,self.harmNo]
        self.x = self.xInd * self.t[-1] / self.t.size
        self.y = self.sound_information.sdInfo['magADSRValue'][:,self.harmNo]
        self.n = self.sound_information.sdInfo['magADSRN'][:,self.harmNo]

        # print(self.x, self.y)

        # draw the points of 2D lines
        self.line = Line2D(self.x, self.y, ls="",
                          marker='o', markerfacecolor='r',
                           animated=True)
        self.canvas.ax.add_line(self.line)
        
        # plot harmonic
        # self.ax.plot(self.t,self.harm)
        self.draw_curve()

        # set the selected point index to None
        self._ind = None
        # set canvas and functions for events
        self.ax_canvas = self.canvas.ax.figure.canvas
        self.ax_canvas.mpl_connect('draw_event', self.draw_callback)
        self.ax_canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.ax_canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.ax_canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        # self.canvas = canvas
        self.canvas.ax.grid()
        self.canvas.draw()

    # draw the curve based on the key points
    def draw_curve(self):
        # sort the key points according to its time
        sort_ind = np.argsort(self.x)
        self.x = self.x[sort_ind]
        self.y = self.y[sort_ind]

        # draw the curve
        nF = self.t.size # the amount of frames
        cv = np.zeros(nF)
        self.xInd = (self.x*nF/self.t[-1]).astype(int)
        points = np.concatenate((np.array([0]), self.xInd, np.array([nF-1])))
        values = np.concatenate((np.array([-150]), self.y, np.array([-150])))
        ts = np.array([self.t]).T

        for j in range(5):
            st_ind = points[j]
            ed_ind = points[j + 1]
            st_v = values[j]
            ed_v = values[j+1]

            if st_ind >= ed_ind - 1:
                if st_ind == ed_ind - 1:
                    cv[st_ind] = st_v
            else:
                cv[st_ind:ed_ind] = UM.curve(self.n[j], np.array([[st_v],[ed_v]]), self.t[st_ind:ed_ind])
        
        cv[-1] = cv[-2]

        # update harmonic
        self.harm = cv

        # plot the curve
        self.plot_curve()


    # curve plotting based on the 
    def plot_curve(self):
        # self.ax.set_xlim((self.t[0]-0.1, self.t[-1]+0.1))
        # self.ax.set_ylim((np.min(self.harm)-2, np.max(self.harm)+2))

        # clear the axes before any new plottings
        for artist in self.canvas.ax.get_children():
            if type(artist).__name__ in ['Line2D']:
                artist.remove()

        # plot the curve
        self.canvas.ax.plot(self.t, self.harm)
        self.canvas.draw()

        # tst
        # print("plot yes!")


    # redraw the canvas
    def draw_callback(self, event):
        self.background = self.ax_canvas.copy_from_bbox(self.canvas.ax.bbox)
        self.canvas.ax.draw_artist(self.line)
        self.ax_canvas.blit(self.canvas.ax.bbox)

    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'
        xt,yt = np.array(self.x),np.array(self.y)
        d = np.sqrt((xt-event.xdata)**2 + (yt-event.ydata)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]
        if d[ind] >=self.offset:
            ind = None
        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts: return
        if event.inaxes==None: return
        if event.button != 1: return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts: return
        if event.button != 1: return
        self._ind = None

        # print(self.x, self.y)
        # draw the new curve and the points
        self.draw_curve()

        self.line = Line2D(self.x, self.y, ls="", marker='o', markerfacecolor='r', animated=True)
        self.canvas.ax.add_line(self.line)

    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showverts: return
        if self._ind is None: return
        if event.inaxes is None: return
        if event.button != 1: return
        x,y = event.xdata, event.ydata
        self.x[self._ind] = x
        self.y[self._ind] = y
        
        # regenerate points while dragging
        self.line = Line2D(self.x, self.y, ls="",
                           marker='o', markerfacecolor='r',
                           animated=True)
        self.canvas.ax.add_line(self.line)
        # restore the canvas
        self.ax_canvas.restore_region(self.background)
        self.canvas.ax.draw_artist(self.line)
        self.ax_canvas.blit(self.canvas.ax.bbox)

if __name__ == '__main__':
    t = np.arange(100)
    harm = np.arange(100)
    init_x = np.array([10,20,40,70])
    init_y = np.array([-20.0,-10.0,-15.5,-30.0])
    init_n = np.array([1,2,3,2,1])
    ax1 = point_drag_axes(t,harm, '1', 0, init_x, init_y, init_n, 0)
    ax2 = point_drag_axes(t,harm, '2', 1, init_x, init_y, init_n, 0)
    # point_drag_fig()
    
