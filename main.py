# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 18:05:07 2021

@author: lucas_lyc
"""

import sys
import pandas as pd
import math
import numpy as np
from numpy import random
from scipy.stats import norm, multivariate_normal, bernoulli, uniform, chi2
from hilbertcurve.hilbertcurve import HilbertCurve
from joblib import Parallel, delayed
import timeit
import ot

a, b, c, d, e, f = sys.argv[1:]
n_dim = int(a) # number of dimensions
n_particles = int(b) # number of particles
n_multiple_des = int(c) # number of decsendants
rho = float(d)# rho
e = float(e) # correlation
target_type = f

from high_dimensional import *
import matplotlib.pyplot as plt

def cov(dim = 2, correlation = 0.5):
    res = np.eye(dim)
    for j in range(dim-1):
        res[j, j+1] = correlation
        res[j+1, j] = correlation
    return res


def log_target_f(t, x):
    if target_type == 'unimodal':
        res = multivariate_normal.pdf(x[0:(t+1)],3*np.ones(t+1),4*cov(t+1, e))
        # return math.log(multivariate_normal.pdf(x[0:(t+1)],3*np.ones(t+1),4*cov(t+1, e)))
    else:
        res = multivariate_normal.pdf(x[0:(t+1)],3*np.ones(t+1),4*cov(t+1, e))+multivariate_normal.pdf(x[0:(t+1)],-3*np.ones(t+1),4*cov(t+1, e))
        # return math.log(multivariate_normal.pdf(x[0:(t+1)],3*np.ones(t+1),4*cov(t+1, e))+multivariate_normal.pdf(x[0:(t+1)],-3*np.ones(t+1),4*cov(t+1, e)))
    if res == 0:
        return float('-inf')
    else:
        return math.log(res)


sd = 3
err_iid = []
err_smg = []
err_more = []
err_ada = []
for _ in range(160):
    Samples_iid, weights_iid, MSE_iid = Sampling(rho = rho, T = n_dim, size = n_particles, multiple_des = n_multiple_des, sd = sd, prop = 'i.i.d.', resample = Hilbert_Resampling, print_step = False)
    Samples_SMG, weights_SMG, MSE_SMG = Sampling(rho = rho, T = n_dim, size = n_particles, multiple_des = n_multiple_des, sd = sd, prop = 'SMG', resample = Hilbert_Resampling, print_step = False)
    Samples_more_particles, weights_more_particles, MSE_more_particles = Sampling(rho= rho, T = n_dim, size = n_multiple_des*n_particles, multiple_des = 1, sd = sd, prop = 'SMG', resample = Hilbert_Resampling, print_step = False)
    Samples_ada, weights_ada, MSE_ada = Adaptive_Sampling(ess_ratio=rho, T = n_dim, size = n_particles*n_multiple_des, multiple_des = 1, sd = sd, prop = 'SMG', resample = Hilbert_Resampling, print_step = False)
    # plotting the last two dimensions; results are random because we need to resample to remove the weights
    # plt.style.use(u'default')
    # plt.rcParams['figure.figsize'] = (15,5)
    # plt.plot(list(range(nT)), list(np.sum((np.array(res_HC[nt]) - np.array(est_ora[nt]))**2) for nt in range(nT)),color='black')
    # plt.plot(list(range(nT)), list(np.sum((np.array(res_OT[nt]) - np.array(est_ora[nt]))**2) for nt in range(nT)),color='blue')
    
    
    #fig = plt.figure(figsize = (15,5))
    #ax1 = fig.add_subplot(1,3,1, adjustable='box', aspect=1, xlim = (-7,7), ylim = (-7,7))
    #ax2 = fig.add_subplot(1,3,2, adjustable='box', aspect=1, xlim = (-7,7), ylim = (-7,7))
    #ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=1, xlim = (-7,7), ylim = (-7,7))
    samples_iid = Hilbert_Resampling(Samples_iid, weights_iid, n_particles, n_dim-1, rho = 1)[0]
    #ax1.scatter(samples_iid[:,n_dim-2],samples_iid[:,n_dim-1])
    #ax1.set_title(str(n_particles)+" particles, "+str(n_multiple_des)+' descendants i.i.d.')
    
    samples_SMG = Hilbert_Resampling(Samples_SMG, weights_SMG, n_particles, n_dim-1, rho= 1)[0]
    #ax2.scatter(samples_SMG[:,n_dim-2],samples_SMG[:,n_dim-1])
    #ax2.set_title(str(n_particles)+" particles, "+str(n_multiple_des)+' descendants SMG')
    
    samples_more = Hilbert_Resampling(Samples_more_particles, weights_more_particles, n_particles, n_dim-1, rho= 1)[0]
    #ax3.scatter(samples_more[:,n_dim-2],samples_more[:,n_dim-1])
    #ax3.set_title(str(n_multiple_des*n_particles)+" particles, 1 descendant")
    #plt.savefig('SMG'+ a +'_'+ b + '_'+ c + '_' + d + '.png',bbox_inches='tight')
    samples_ada = Hilbert_Resampling(Samples_ada, weights_ada, n_particles, n_dim-1, rho= 1)[0]
    
    component = bernoulli.rvs(p = 1/2, size = n_particles)
    multi_sample = multivariate_normal.rvs(mean = np.zeros(n_dim), cov = np.eye(n_dim), size = n_particles)
    sample_oracle = np.array([component[i]*np.ones(n_dim) + multi_sample[i] for i in range(n_particles)])
    
    eqweight = np.ones(n_particles)/n_particles
    dist_mat = np.array([[np.linalg.norm(sample_oracle[i] - samples_iid[j]) for j in range(n_particles)] for i in range(n_particles)])
    err1= ot.emd2(eqweight, eqweight, dist_mat)
    
    eqweight = np.ones(n_particles)/n_particles
    dist_mat = np.array([[np.linalg.norm(sample_oracle[i] - samples_SMG[j]) for j in range(n_particles)] for i in range(n_particles)])
    err2= ot.emd2(eqweight, eqweight, dist_mat)
    
    eqweight = np.ones(n_particles)/n_particles
    dist_mat = np.array([[np.linalg.norm(sample_oracle[i] - samples_more[j]) for j in range(n_particles)] for i in range(n_particles)])
    err3= ot.emd2(eqweight, eqweight, dist_mat)
    
    eqweight = np.ones(n_particles)/n_particles
    dist_mat = np.array([[np.linalg.norm(sample_oracle[i] - samples_ada[j]) for j in range(n_particles)] for i in range(n_particles)])
    err4= ot.emd2(eqweight, eqweight, dist_mat)
    
    err_iid.append(err1)
    err_smg.append(err2)
    err_more.append(err3)
    err_ada.append(err4)
res = pd.DataFrame([err_iid, err_smg, err_more, err_ada])
res.to_csv(str(n_dim) + '_' + str(correlation) + '_' + target_type + '_' + str(n_particles) + '_' + str(n_multiple_des) + '_' + str(rho) + 'res.csv', index = False)
# =============================================================================
# f= open('weighted_resampling.csv', 'a')
# f.write(a + ',' + ',' + b + ','+ c + ','+ d + ',' + str(err1) + ',' + str(err2) + ',' + str(err3) + '\n')
# f.close()
# =============================================================================
