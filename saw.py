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


a, b, c, d, e = sys.argv[1:]
n_dim = int(a) # number of dimensions
n_particles = int(b) # number of particles
rho = float(d)# rho
ess_ratio = float(e) # ess ratio



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
    resampling_weights_sum = np.sum(resampling_weights)
    if rho == 1:
        weights_after = np.ones(len(weights))/len(weights)
    else:
        weights_after = np.power(weights, 1-rho)
    resampling_weights = resampling_weights/resampling_weights_sum
    weights_after = weights_after/np.sum(weights_after)
    return resampling_weights, weights_after

def Stratified_Resampling(particles, weights, size, rho):
    # Wenshuo: now returns weighted particles
    resampling_weights, weights_after = General_Resampling_Weights(weights, rho)
    weight_matrix = Stratified_Matrix(resampling_weights, M = size)
    indices = [random.choice(range(len(particles)), p = w) for w in weight_matrix]
    res = [np.array([particles[i] for i in indices]), np.array([weights_after[i] for i in indices])]
    return res


def Multinomial_Resampling(particles, weights, size, rho):
    resampling_weights, weights_after = General_Resampling_Weights(weights, rho)
    indices = [random.choice(range(len(particles)), p = resampling_weights) for _ in range(size)]
    xx = [particles[i] for i in indices]
    ww = np.array([weights_after[i] for i in indices])
    ww = ww/np.sum(ww)
    return xx, ww

def Multiple_Descendent_Proposal(particles, weights, t, descendant = 'stratified'):
    x_prop = []
    weight_prop = []
    i = 0
    for xi in particles:
        up = np.array(xi[t-1]) + np.array([0,1])
        down = np.array(xi[t-1]) + np.array([0,-1])
        left = np.array(xi[t-1]) + np.array([-1,0])
        right = np.array(xi[t-1]) + np.array([1,0])
        legal = []
        for possible_xt in [up, down, left, right]:
            if tuple(possible_xt) not in [tuple(xx) for xx in xi]:
                legal.append(xi + [possible_xt])
        if len(legal) > 0:
            k = len(legal)
            weight_prop = weight_prop + [k*weights[i]]*k
            if descendant == 'stratified':
                x_prop = x_prop + legal
            else:
                indices = [random.choice(range(k), p = np.array([1/k for _ in range(k)])) for w in range(k)]
                x_prop = x_prop + [legal[j] for j in indices]
        i = i + 1
    return x_prop, weight_prop

def Random_Walk_Proposal(particles, weights, t):
    x_prop = []
    weight_prop = []
    i = 0
    for xi in particles:
        up = np.array(xi[t-1]) + np.array([0,1])
        down = np.array(xi[t-1]) + np.array([0,-1])
        left = np.array(xi[t-1]) + np.array([-1,0])
        right = np.array(xi[t-1]) + np.array([1,0])
        legal = []
        possible_xt = [up, down, left, right][random.choice(4)]
        if tuple(possible_xt) not in [tuple(xx) for xx in xi]:
            legal.append(xi + [possible_xt])
        if len(legal) > 0:
            k = len(legal)
            weight_prop = weight_prop + [k*weights[i]]*k
            x_prop = x_prop + legal
        i = i + 1
    return x_prop, weight_prop

def Sampling(rho, ess_ratio = 1, T = 10, size = 100,  prop = 'stratified', resample = Multinomial_Resampling, print_step = True): #need modification
    # change ess_ratio for adaptive resampling
    if print_step:
        print("dimension "+ str(1) + "/" + str(T))
    w = np.ones(size)/size
    xt1 = [[np.array([0,0])] for _ in range(size)]
    normalizing_constant_estimate = [1.0]*T
    log_nomalizing_constant_estimate = 0
    if prop == 'stratified':
        for t in range(1,T):
            if print_step:
                print("dimension "+ str(t+1) + "/" + str(T))
            xt1, w = Multiple_Descendent_Proposal(xt1, w, t, prop)
            normalizing_constant_estimate[t] = np.mean(w)
            w = w/np.sum(w)
            if t<T-1:
                log_nomalizing_constant_estimate += np.log(np.mean(w)) 
                xt1, w = resample(xt1, w, size, rho)
            if t==T-1:
                log_nomalizing_constant_estimate += np.log(np.mean(w)) 
    else:
        for t in range(1,T):
            if print_step:
                print("dimension "+ str(t+1) + "/" + str(T))
            xt1, w = Random_Walk_Proposal(xt1, w, t)
            normalizing_constant_estimate[t] = np.mean(w)
            w = w/np.sum(w)
            log_nomalizing_constant_estimate += np.log(np.mean(w)) 
            if t<T-1 and 1/sum(w**2) < ess_ratio*len(w):
                xt1, w = resample(xt1, w, size, rho)
   
    return xt1, w, log_nomalizing_constant_estimate

 
res_log_normal = []
res_log_normal_rw = []
for _ in range(160):
    Samples_iid, weights_iid, log_normal = Sampling(rho = rho, ess_ratio = ess_ratio ,T = n_dim, size = n_particles, print_step = True)
    Samples_rw, weights_rw, log_normal_rw = Sampling(rho = rho, ess_ratio = ess_ratio ,T = n_dim, size = n_particles, prop = 'rw', print_step = True)

res_log_normal.append(log_normal)
res_log_normal_rw.append(log_normal_rw)

res = pd.DataFrame([res_log_normal, res_log_normal_rw])
res.to_csv('/n/jun_liu_lab/saw/' + str(n_particles) + '_' + str(rho) + '_' + str(ess_ratio) +'res.csv', index = False)
# =============================================================================
# f= open('weighted_resampling.csv', 'a')
# f.write(a + ',' + ',' + b + ','+ c + ','+ d + ',' + str(err1) + ',' + str(err2) + ',' + str(err3) + '\n')
# f.close()
# =============================================================================
