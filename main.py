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
correlation = float(e) # correlation
target_type = f



def cov(dim = 2, correlation = 0.5):
    res = np.eye(dim)
    for j in range(dim-1):
        res[j, j+1] = correlation
        res[j+1, j] = correlation
    return res


def log_target_f(t, x):
    if target_type == 'unimodal':
        res = multivariate_normal.pdf(x[0:(t+1)],3*np.ones(t+1),4*cov(t+1, correlation))
    else:
        res = multivariate_normal.pdf(x[0:(t+1)],3*np.ones(t+1),4*cov(t+1, correlation))+multivariate_normal.pdf(x[0:(t+1)],-3*np.ones(t+1),4*cov(t+1, correlation))
    if res == 0:
        return float('-inf')
    else:
        return math.log(res)

def oracle_sampling(ndim):
    if target_type == 'unimodal':
        return multivariate_normalrvs(mean = 3*np.ones(ndim), cov = 4*cov(ndim, correlation), size = 1)
    else:
        return bernoulli.rvs(p = 1/2, size = 1) * multivariate_normal.rvs(mean = 3*np.ones(ndim), cov = 4*cov(ndim, correlation), size = 1)

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:33:14 2020

@author: Wenshuo Wang
"""
import pandas as pd
import math
import numpy as np
from numpy import random
from scipy.stats import norm, multivariate_normal, uniform, chi2
from hilbertcurve.hilbertcurve import HilbertCurve
from joblib import Parallel, delayed
import timeit
import sys

# def cov(dim = 2, correlation = 0.5):
#     res = np.eye(dim)
#     for j in range(dim-1):
#         res[j, j+1] = correlation
#         res[j+1, j] = correlation
#     return res

# def log_target_f(t, x):
#     return(math.log(multivariate_normal.pdf(x[0:(t+1)],3*np.ones(t+1),4*cov(t+1, 0.5))+multivariate_normal.pdf(x[0:(t+1)],-3*np.ones(t+1),4*cov(t+1, 0.5))))

def Stratified_Matrix(ww, M):
    w = ww.copy()
    size = len(ww)
    i = 0
    j = 0
    weight_matrix_res = np.zeros([M,size])
    cumsum = np.zeros(M)
    while i<M and j<size:
        if (w[j]<1/M):
            if ((cumsum[i]+w[j] * M) <= 1):
                weight_matrix_res[i,j] = w[j] * M
                cumsum[i]= cumsum[i] + w[j] * M
                j = j + 1
            else:
                weight_matrix_res[i,j] =1-cumsum[i]
                w[j] = w[j] - weight_matrix_res[i,j]/M
                i = i + 1
        else:
            weight_matrix_res[i,j] = 1-cumsum[i]
            w[j] = w[j] - weight_matrix_res[i,j]/M
            i = i + 1
    return weight_matrix_res

def General_Resampling_Weights(weights, rho):
    resampling_weights = np.power(weights, rho)
    if rho == 1:
        weights_after = np.ones(len(weights))
    else:
        weights_after = np.power(weights, 1-rho)
    resampling_weights = resampling_weights/np.sum(resampling_weights)
    weights_after = weights_after/np.sum(weights_after)
    return resampling_weights, weights_after

def Stratified_Resampling(particles, weights, size, rho):
    # Wenshuo: now returns weighted particles
    resampling_weights, weights_after = General_Resampling_Weights(weights, rho)
    weight_matrix = Stratified_Matrix(resampling_weights, M = size)
    indices = [random.choice(range(len(particles)), p = w) for w in weight_matrix]
    res = [np.array([particles[i] for i in indices]), np.array([weights_after[i] for i in indices])]
    return res

def Hilbert_Resampling(particles, weights, size, t, rho):
    particles = np.array(particles)
    # print(particles)
    dim = particles.shape[1]
    dim = t+1
    pmax = [max(particles[:,k])+0.1 for k in range(dim)]
    pmin = [min(particles[:,k])-0.1 for k in range(dim)]
    unified_particles = np.array([[(par[k]-pmin[k])/(pmax[k]-pmin[k]) for k in range(dim)] for par in particles])
    hilbert_mapping = [Hilbert_Mapping(up, dim=dim) for up in unified_particles]
    # getting weights
    resampling_weights, weights_after = General_Resampling_Weights(weights, rho)
    Weighted_Sample = pd.concat([pd.DataFrame(particles),pd.DataFrame({"weight": resampling_weights, 'map': hilbert_mapping})],axis=1)
    Weighted_Sample = Weighted_Sample.sort_values(by = ['map'], ascending = True)
    Weighted_Sample.index = range(Weighted_Sample.shape[0])
    w = Weighted_Sample['weight']
    weight_matrix = Stratified_Matrix(w, M = size)
    
    indices = [random.choice(range(len(particles)), p = w) for w in weight_matrix]
    xx = np.array(Weighted_Sample.iloc[indices, list(range(particles.shape[1]))])
    ww = np.array([weights_after[i] for i in indices])
    return xx, ww

def Multinomial_Resampling(particles, weights, size, rho):
    resampling_weights, weights_after = General_Resampling_Weights(weights, rho)
    indices = [random.choice(range(len(particles)), p = weights) for _ in range(size)]
    xx = np.array([particles[i] for i in indices])
    ww = np.array([weights_after[i] for i in indices])
    return xx, ww

def Multiple_Descendent_Proposal(particles, weights, t, multiple_des = 4, sd = 3):
    T = particles.shape[1]
    size = particles.shape[0]
    x_prop = np.zeros((size*multiple_des,T))
    weight = np.zeros(size*multiple_des)
    for ipar in range(size):
        for k in range(multiple_des):
            x_prop[ipar*multiple_des+k] = particles[ipar]
            x_prop[ipar*multiple_des+k,t] = random.normal(0,sd)
            weight[ipar*multiple_des+k] = log_target_f(t, x_prop[ipar*multiple_des+k]) - log_target_f(t-1, particles[ipar]) - norm.logpdf(x_prop[ipar*multiple_des+k,t], 0, sd)
    x_prop = np.array(x_prop)
    weight = weight - np.max(weight)
    weight = np.exp(weight)
    weight = weight/np.sum(weight)
    for ipar in range(size):
        for k in range(multiple_des):
            weight[ipar*multiple_des+k] = weight[ipar*multiple_des+k] * weights[ipar]
    weight = weight/np.sum(weight)
    return x_prop, weight

def Hilbert_Mapping(x, p = 8, dim = 10):
    hilbert_curve = HilbertCurve(p, dim)
    aa = [int(xx) for xx in x*2**p]
    h = hilbert_curve.distance_from_coordinates(aa)
    h = np.array(h/(2**(p*dim)))
    return h

def Hilbert_Mapping_Inverse(h, p = 8, dim = 10):
    hilbert_curve = HilbertCurve(p, dim)
    aa = hilbert_curve.coordinates_from_distance(int(h*2**(p*dim)))
    aa = np.array(aa) + 0.5
    aa = aa/(2**p)
    return aa

def Hilbert_Stratified_Proposal(particles, weights, t, multiple_des = 4, sd = 3):
    # not done, add weights and rho; see Multiple_Descendent_Proposal
    T = particles.shape[1]
    size = particles.shape[0]
    x_prop = np.zeros((size*multiple_des,T))
    weight = np.zeros(size*multiple_des)
    for ipar in range(size):
        hh = [(uniform.rvs() + md)/multiple_des for md in range(multiple_des)]
        vv = [Hilbert_Mapping_Inverse(h, dim = 1) for h in hh]
        xx = [sd*norm.ppf(v) for v in vv]
        for k in range(multiple_des):
            x_prop[ipar*multiple_des+k] = particles[ipar]
            x_prop[ipar*multiple_des+k,t] = xx[k]
            weight[ipar*multiple_des+k] = log_target_f(t, x_prop[ipar*multiple_des+k]) - log_target_f(t-1, particles[ipar]) - norm.logpdf(x_prop[ipar*multiple_des+k,t], 0, sd)
    x_prop = np.array(x_prop)
    weight = weight - np.max(weight)
    weight = np.exp(weight)
    weight = weight/np.sum(weight)
    for ipar in range(size):
        for k in range(multiple_des):
            weight[ipar*multiple_des+k] = weight[ipar*multiple_des+k] * weights[ipar]
    weight = weight/np.sum(weight)
    return x_prop, weight


def Sampling(rho, T = 10, size = 100, multiple_des = 4, sd = 3, prop = 'i.i.d.', resample = Hilbert_Resampling, print_step = True): #need modification
    if print_step:
        print("dimension "+ str(1) + "/" + str(T))
    w = np.zeros(size)
    xt1 = np.zeros((size,T))
    xt1[:,0] = np.array([random.normal(0,sd) for _ in range(size)])
    for i in range(size):
        w[i] = log_target_f(0,xt1[i]) - norm.logpdf(xt1[i,0], 0, sd)
    w = [x-max(w) for x in w]
    w = [math.exp(x) for x in w]
    w = [x/sum(w) for x in w]
    w = np.array(w)
    xt1, w = resample(xt1, w, size, 0, rho)

    for t in range(1,T):
        if print_step:
            print("dimension "+ str(t+1) + "/" + str(T))
        if prop == 'i.i.d.':
            xt1star, w = Multiple_Descendent_Proposal(xt1, w, t, multiple_des, sd)
        elif prop == 'SMG':
            xt1star, w = Hilbert_Stratified_Proposal(xt1, w, t, multiple_des, sd)
        if t<T-1:
            xt1, w = resample(xt1star, w, size, t, rho)
        if t==T-1:
            return xt1star, w, np.linalg.norm(np.transpose(xt1star)@w)**2
            # add variance of each step

def Adaptive_Sampling(ess_ratio, T = 10, size = 100, multiple_des = 4, sd = 3, prop = 'i.i.d.', resample = Hilbert_Resampling, print_step = True): #need modification
    if print_step:
        print("dimension "+ str(1) + "/" + str(T))
    w = np.zeros(size)
    xt1 = np.zeros((size,T))
    xt1[:,0] = np.array([random.normal(0,sd) for _ in range(size)])
    for i in range(size):
        w[i] = log_target_f(0,xt1[i]) - norm.logpdf(xt1[i,0], 0, sd)
    w = [x-max(w) for x in w]
    w = [math.exp(x) for x in w]
    w = [x/sum(w) for x in w]
    w = np.array(w)
    xt1, w = resample(xt1, w, size, 0, 1)

    for t in range(1,T):
        if print_step:
            print("dimension "+ str(t+1) + "/" + str(T))
        if prop == 'i.i.d.':
            xt1star, w = Multiple_Descendent_Proposal(xt1, w, t, multiple_des, sd)
        elif prop == 'SMG':
            xt1star, w = Hilbert_Stratified_Proposal(xt1, w, t, multiple_des, sd)
        if t<T-1 and 1/sum(w**2) < ess_ratio*size*multiple_des:
            xt1, w = resample(xt1star, w, size, t, 1)
        if t==T-1:
            return xt1star, w, np.linalg.norm(np.transpose(xt1star)@w)**2
    
import matplotlib.pyplot as plt

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
    
    # component = bernoulli.rvs(p = 1/2, size = n_particles)
    # multi_sample = multivariate_normal.rvs(mean = np.zeros(n_dim), cov = np.eye(n_dim), size = n_particles)
    sample_oracle = np.array([oracle_sampling(n_dim) for i in range(n_particles)])
    
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
res.to_csv('/n/jun_liu_lab/smg/' + str(n_dim) + '_' + str(correlation) + '_' + target_type + '_' + str(n_particles) + '_' + str(n_multiple_des) + '_' + str(rho) + 'res.csv', index = False)
# =============================================================================
# f= open('weighted_resampling.csv', 'a')
# f.write(a + ',' + ',' + b + ','+ c + ','+ d + ',' + str(err1) + ',' + str(err2) + ',' + str(err3) + '\n')
# f.close()
# =============================================================================
