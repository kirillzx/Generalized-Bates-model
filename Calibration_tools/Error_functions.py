import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.integrate import quad
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from functools import partial
from scipy.optimize import minimize, fsolve, basinhopping, newton, dual_annealing
import tqdm
import yfinance as yf
import yahoo_fin.options as ops
from datetime import datetime
import cmath

import Functions.exact_methods as exm
import Functions.characteristics_functions as chf
import Functions.monte_carlo_methods as mcm
import Functions.stochastic_processes as stch

def errorFun_DCL(CP, tau, T, k, gamma, vb, kr, gammar, mur, krho, delta, rho4, rho5, xip, muJ, sigmaJ, v0, r0, rho0, K, marketPrice, s0, N, L):
    if K is not np.array:
        K = np.array(K).reshape([len(K),1])

   
    cf = chf.ChFBates_StochIR_StochCor_DCL(tau, T, k, gamma, vb, kr, gammar, mur, krho, delta, rho4, rho5, xip, muJ, sigmaJ, v0, r0, rho0, np.log(s0))
    valCOS = exm.optionPriceCOSMthd_StochIR(cf, CP, s0, T, K, N, L)

    errorVector = np.mean((valCOS.T[0] - marketPrice)**2)

    return errorVector


def calibrationBates_SIR_SC_DCL(CP, K, marketPrice, s0, T, tau, N, L, method='BFGS'):
    K = np.array(K)
    marketPrice = np.array(marketPrice)
   
    # k, gamma, vb, kr, gammar, mur, krho, delta, rho4, rho5, xip, muJ, sigmaJ, v0, r0, rho0
    f_obj = lambda x: errorFun_DCL(CP, tau, T, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],
                               x[10], x[11], x[12], x[13], x[14], x[15], K, marketPrice, s0, N, L)

                        # k, gamma, vb, kr, gammar, mur, theta, delta, rho4, rho5, xip, muJ, sigmaJ, v0, r0, rho0
    initial = np.array([0.4, 0.3, 0.04, 0.3, 0.2,   0.2,   2,   1,  0.1,  0.5, 1,      0.1,    1,   0.04, 0.05, 0.7])

            # k,    gamma,   vb,     kr,   gammar, mur, theta, delta, rho4, rho5,  xip,    muJ, sigmaJ, v0, r0, rho0
    xmin = [0.01,   0.01,   0.001, 0.05,   0.01,  0.001, 0.01,    1,   -0.98, -0.98, 0.01, -5,   0.01,  0.01,   0.01, -0.98]
    xmax = [2,       0.7,   1,     2,      0.3,     0.1,    5,    4,   0.98, 0.98,  5,     5,   10,      1, 0.2, 0.98] 
    bounds = [(low, high) for low, high in zip(xmin, xmax)]
    
    pars = minimize(f_obj, x0=initial, tol=1e-5, bounds=bounds, options = {'maxiter': 500, 'disp':False}, method=method)
    # pars = dual_annealing(f_obj, x0=initial, bounds=bounds, maxiter=200)

    print(pars)
    
    parmCalibr =  {"k":pars.x[0], "gamma": pars.x[1], "vb": pars.x[2], "kr": pars.x[3], "gammar": pars.x[4], "mur":pars.x[5],\
                   "theta": pars.x[6], "delta": pars.x[7], "rho4":pars.x[8], "rho5": pars.x[9], "xi": pars.x[10], "muJ": pars.x[11],\
                    "sigmaJ": pars.x[12], "v0":pars.x[13], "r0": pars.x[14], "rho0": pars.x[15], 'ErrorFinal': pars.fun}
    
    return parmCalibr



def errorFun_OU(CP, tau, T, k, gamma, vb, kr, gammar, mur, krho, delta, rho4, rho5, xip, muJ, sigmaJ, v0, r0, rho0, K, marketPrice, s0, N, L, mu_rho, sigma_rho):
    if K is not np.array:
        K = np.array(K).reshape([len(K),1])

   
    cf = chf.ChFBates_StochIR_StochCor_OU(tau, T, k, gamma, vb, kr, gammar, mur, krho, delta, rho4, rho5, xip, muJ, sigmaJ,
                                          v0, r0, rho0, np.log(s0), mu_rho, sigma_rho)
    valCOS = exm.optionPriceCOSMthd_StochIR(cf, CP, s0, T, K, N, L)

    errorVector = np.mean((valCOS.T[0] - marketPrice)**2)

    return errorVector

def calibrationBates_SIR_SC_OU(CP, K, marketPrice, s0, T, tau, N, L, method='BFGS'):
    K = np.array(K)
    marketPrice = np.array(marketPrice)
   
    # k, gamma, vb, kr, gammar, mur, krho, delta, rho4, rho5, xip, muJ, sigmaJ, v0, r0, rho0
    f_obj = lambda x: errorFun_OU(CP, tau, T, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],
                               x[10], x[11], x[12], x[13], x[14], x[15], K, marketPrice, s0, N, L, x[16], x[17])

                        # k, gamma, vb, kr, gammar, mur, theta, delta, rho4, rho5, xip, muJ, sigmaJ, v0, r0, rho0
    initial = np.array([0.4, 0.3, 0.04, 0.3, 0.2,   0.2,   2,   1,  0.1,  0.5, 1,      0.1,    1,   0.04, 0.05, 0.7, 0.1, 0.3])

            # k,    gamma,   vb,     kr,   gammar, mur, theta, delta, rho4, rho5,  xip,    muJ, sigmaJ, v0, r0, rho0
    xmin = [0.0001, 0.0001, 0.001, 0.05, 0.01,    0.001, 1,    1,  -0.98, -0.98, 0.001, -10,   0.01,  0.0001, 0.0001, -0.98, -0.7, 0.001]
    xmax = [5,       0.7,   1,     2,      0.3,     0.1,    5,    4,   0.98, 0.98,  10,     10,   3,      1, 0.2, 0.98, 0.7, 0.999] 
    bounds = [(low, high) for low, high in zip(xmin, xmax)]
    
    pars = minimize(f_obj, x0=initial, tol=1e-5, bounds=bounds, options = {'maxiter': 500, 'disp':False}, method=method)
    # pars = dual_annealing(f_obj, x0=initial, bounds=bounds, maxiter=200)

    print(pars)
    
    parmCalibr =  {"k":pars.x[0], "gamma": pars.x[1], "vb": pars.x[2], "kr": pars.x[3], "gammar": pars.x[4], "mur":pars.x[5],\
                   "theta": pars.x[6], "delta": pars.x[7], "rho4":pars.x[8], "rho5": pars.x[9], "xi": pars.x[10], "muJ": pars.x[11],\
                    "sigmaJ": pars.x[12], "v0":pars.x[13], "r0": pars.x[14], "rho0": pars.x[15], "mu_rho": pars.x[16], "sigma_rho": pars.x[17],\
                    'ErrorFinal': pars.fun}
    
    return parmCalibr




def errorFun_bates(CP, tau, T, kappa,gamma,vbar,v0,rho,xiP,muJ,sigmaJ, K, marketPrice, s0, N, L, r):
    if K is not np.array:
        K = np.array(K).reshape([len(K),1])

   
    cf = chf.ChFBatesModel(tau, T, kappa,gamma,vbar,v0,rho,xiP,muJ,sigmaJ)
    valCOS = exm.CallPutOptionPriceCOS(cf, CP, s0, r, tau, K, N, L)

    errorVector = np.mean((valCOS.T[0] - marketPrice)**2)

    return errorVector

def calibrationBates(CP, K, marketPrice, s0, T, tau, N, L, r):
    K = np.array(K)
    marketPrice = np.array(marketPrice)
   
    
    f_obj = lambda x: errorFun_bates(CP, tau, T, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], K, marketPrice, s0, N, L, r)

                        # k, gamma, vb,
    initial = np.array([0.4, 0.3, 0.04, 0.1, 0.1, 1, 0.01, 0.4])

            # k,    gamma,   vb,   
    xmin = [0.0001, 0.0001, 0.001, 0.001, -0.999, 0.0001, -10, 0.0001]
    xmax = [10,       3,   3,   0.3, 0.999, 10, 10, 3] 
    bounds = [(low, high) for low, high in zip(xmin, xmax)]
    
    pars = minimize(f_obj, x0=initial, tol=1e-5, bounds=bounds, options = {'maxiter': 200, 'disp':False})
    # pars = dual_annealing(f_obj, x0=initial, bounds=bounds, maxiter=200)

    print(pars)
    
    parmCalibr =  {"k": pars.x[0], "gamma": pars.x[1], "vb": pars.x[2], "v0": pars.x[3], "rho": pars.x[4], "xi":pars.x[5],\
                   "muJ": pars.x[6], "sigmaJ": pars.x[7],\
                    'ErrorFinal': pars.fun}
    
    return parmCalibr