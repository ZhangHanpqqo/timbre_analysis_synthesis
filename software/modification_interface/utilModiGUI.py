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

class point_drag_axes:
    showverts = True
    offset = 5 # 距离偏差设置

    def __init__(self, t, harm, harmName, harmNo, init_xInd, init_y, init_ns=np.array([1,1,1,1,1]), init_variance=0):
        # set an axes
        self.ax = plt.axes()
        # set title
        self.ax.set_title('Harmonic #'+str(harmNo)+' time-'+harmName)
        # set limitations
        self.ax.set_xlim((t[0]-0.1, t[-1]+0.1))
        self.ax.set_ylim((-155, 5))
        # set labels
        self.ax.set_xlabel('time(s)')
        self.ax.set_ylabel(harmName)

        # set initial parameters
        self.harm = harm
        self.harmName = harmName
        self.xInd = init_xInd
        self.t = t
        self.x = self.xInd * self.t[-1] / self.t.size
        self.y = init_y
        self.n = init_ns
        self.var = init_variance

        # draw the points of 2D lines
        self.line = Line2D(self.x, self.y, ls="",
                           marker='o', markerfacecolor='r',
                           animated=True)
        self.ax.add_line(self.line)
        
        # plot harmonic
        # self.ax.plot(self.t,self.harm)
        self.draw_curve()

        # set the selected point index to None
        self._ind = None
        # set canvas and functions for events
        self.canvas = self.ax.figure.canvas
        self.canvas.mpl_connect('draw_event', self.draw_callback)
        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        # self.canvas = canvas
        plt.grid()
        plt.show()

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
        if self.harmName == 'magnitude':
            values = np.concatenate((np.array([-150]), self.y, np.array([-150])))
        else:
            print(self.y)
            values = np.concatenate((self.y[0:1],self.y,self.y[-1:]))
        
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

        # add variance
        if self.var != 0:
            cv = cv + UM.smooth(np.random.normal(0,self.var,nF),31)

        # update harm
        self.harm = cv

        # plot the curve
        self.plot_curve()


    # curve plotting based on the 
    def plot_curve(self):
        self.ax.set_xlim((self.t[0]-0.1, self.t[-1]+0.1))
        self.ax.set_ylim((np.min(self.harm)-2, np.max(self.harm)+2))

        # clear the axes before any new plottings
        for artist in self.ax.get_children():
            if type(artist).__name__ in ['Line2D']:
                artist.remove()

        # plot the curve
        self.ax.plot(self.t, self.harm)

        # tst
        print("plot yes!")


    # redraw the canvas
    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

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

        # draw the new curve and the points
        self.draw_curve()

        self.line = Line2D(self.x, self.y, ls="", marker='o', markerfacecolor='r', animated=True)
        self.ax.add_line(self.line)

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
        self.ax.add_line(self.line)
        # restore the canvas
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


# class point_drag_fig:
#     def __init__(self):
#         self.fig, self.axes = plt.subplots(1,2)
# 
#         t = np.arange(100)
#         harm = np.arange(100)
#         init_x = np.array([10,20,40,70])
#         init_y = np.array([-20.0,-10.0,-15.5,-30.0])
#         init_n = np.array([1,2,3,2,1])
#         self.axes[0] = point_drag_axes(t,harm, '1', 0, init_x, init_y, init_n, 0).ax
#         self.axes[1] = point_drag_axes(t,harm, '2', 1, init_x, init_y, init_n, 0).ax
#         self.fig.show()
#         plt.pause(100)
        

if __name__ == '__main__':
    t = np.arange(100)
    harm = np.arange(100)
    init_x = np.array([10,20,40,70])
    init_y = np.array([-20.0,-10.0,-15.5,-30.0])
    init_n = np.array([1,2,3,2,1])
    ax1 = point_drag_axes(t,harm, '1', 0, init_x, init_y, init_n, 0)
    ax2 = point_drag_axes(t,harm, '2', 1, init_x, init_y, init_n, 0)
    # point_drag_fig()
    
