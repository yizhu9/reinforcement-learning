"""
==================================
Inverted pendulum animation class
==================================

Adapted from the double pendulum problem animation.
https://matplotlib.org/examples/animation/double_pendulum_animated.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class EnvAnimate:

    '''
    Initialize Inverted Pendulum
    '''
    def __init__(self):

        # Change this to match your discretization
        self.dt = 0.05
        self.t = np.arange(0.0, 2.0, self.dt)

        # Random trajectory for example
        self.theta = np.linspace(-np.pi, np.pi, self.t.shape[0])
        self.x1 = np.sin(self.theta)
        self.y1 = np.cos(self.theta)
        self.u = np.zeros(self.t.shape[0])

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2, -2, 2])

        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.8, '', transform=self.ax.transAxes)

    '''
    Provide new rollout theta values to reanimate
    '''
    def new_data(self, theta):
        self.theta = theta
        self.x1 = np.sin(theta)
        self.y1 = np.cos(theta)
        #self.u = np.zeros(t.shape[0])

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2,-2, 2])
        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)

    def init(self):
        self.line.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.time_text

    def _update(self, i):
        thisx = [0, self.x1[i]]
        thisy = [0, self.y1[i]]
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (self.t[i], self.theta[i], self.u[i]))
        return self.line, self.time_text

    def start(self):
        print('Starting Animation')
        print()
        # Set up plot to call animate() function periodically
        self.ani = FuncAnimation(self.fig, self._update, frames=range(len(self.x1)), interval=25, blit=True, init_func=self.init, repeat=False)
        plt.show()


if __name__ == '__main__':

	animation = EnvAnimate()
	animation.start()