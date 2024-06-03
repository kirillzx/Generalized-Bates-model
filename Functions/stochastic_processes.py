import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.integrate import quad
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from functools import partial
from scipy.optimize import minimize, fsolve, newton
import matplotlib.pyplot as plt 

class StochasticProcesses:
    def __init__(self, option_type='call'):
        self.option_type = option_type

    def geometric_BM(self, T, s0, mu, sigma, n):
        x = np.zeros(n + 1)
        s = np.zeros(n + 1)
        time = np.zeros(n + 1)

        x[0] = np.log(s0)
        dt = T/float(n)

        for t in range(n):
            x[t+1] = x[t] + (mu - (sigma**2)/2) * dt + sigma * np.sqrt(dt) * stats.norm.rvs(loc=0, scale=1)
            time[t+1] = time[t] + dt

        s = np.exp(x)

        return time, s

    def poisson_process(self, T, s0, xiP, n, type_p='ordinary'):
        time = np.zeros(n + 1)
        dt = T/float(n)
        poisson_distr = np.random.poisson(xiP * dt, n + 1)

        if type_p == 'ordinary':
            x = np.zeros(n + 1)
            x[0] = s0

            for t in range(n):
                x[t+1] = x[t] + poisson_distr[t]
                time[t+1] = time[t] + dt

            return time, x

        elif type_p == 'compensated':
            xc = np.zeros(n + 1)
            xc[0] = s0

            for t in range(n):
                xc[t+1] = xc[t] + poisson_distr[t] - xiP * dt
                time[t+1] = time[t] + dt

            return time, xc
        else:
            print('Wrong type process, available: ordinary and compensated')

    def merton_process(self, T, s0, xiP, muj, sigmaj, r, sigma, n):
        time = np.zeros(n + 1)
        dt = T/float(n)

        z = np.random.normal(0.0, 1.0, n + 1)
        zj = np.random.normal(muj, sigmaj, n + 1)
        poisson_distr = np.random.poisson(xiP * dt, n + 1)

        x = np.zeros(n + 1)
        s = np.zeros(n + 1)

        s[0] = s0
        x[0] = np.log(s0)

        EeJ = np.exp(muj + 0.5 * sigmaj**2)

        for t in range(n):
            x[t+1] = x[t] + (r - xiP * (EeJ - 1) - 0.5 * sigma**2) * dt +\
                sigma * np.sqrt(dt) * z[t] + zj[t] * poisson_distr[t]

            time[t+1] = time[t] + dt

        s = np.exp(x)

        return time, s





    def plot_path(self, time, s, label='Process path', title='Process Visualization'):
        plt.subplots(figsize=(10, 5), dpi=100)
        plt.plot(time, s, label=label)

        plt.title(title, fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Values', fontsize=14)
        plt.legend()
        plt.show()
