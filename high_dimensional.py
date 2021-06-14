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

# def log_target_f(t, x):
#     return(math.log(multivariate_normal.pdf(x[0:(t+1)],3*np.ones(t+1),2*np.eye(t+1))+multivariate_normal.pdf(x[0:(t+1)],-3*np.ones(t+1),2*np.eye(t+1))))

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
    res = [np.array(Weighted_Sample.iloc[indices, list(range(particles.shape[1]))]), np.array([weights_after[i] for i in indices])]
    return res

def Multinomial_Resampling(particles, weights, size, rho):
    resampling_weights, weights_after = General_Resampling_Weights(weights, rho)
    indices = [random.choice(range(len(particles)), p = weights) for _ in range(size)]
    res = [np.array([particles[i] for i in indices]), np.array([weights_after[i] for i in indices])]
    return res

def Multiple_Descendent_Proposal(particles, weights, rho, t, multiple_des = 4, sd = 3):
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
    weight = np.array([weight[i]*weights[i] for i in range(size)])
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

def Hilbert_Stratified_Proposal(particles, weights, rho, t, multiple_des = 4, sd = 3):
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
    return x_prop, weight

def Sampling(T = 10, size = 100, multiple_des = 4, sd = 3, prop = 'i.i.d.', resample = Hilbert_Resampling, print_step = False, alpha): # need modification
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
    xt1 = Hilbert_Resampling(xt1, w, size, 0)
    for t in range(1,T):
        if print_step:
            print("dimension "+ str(t+1) + "/" + str(T))
        if prop == 'i.i.d.':
            xt1star, w = Multiple_Descendent_Proposal(xt1, w, rho, t, multiple_des, sd)
        elif prop == 'SMG':
            xt1star, w = Hilbert_Stratified_Proposal(xt1, w, rho, t, multiple_des, sd)
        if t<T-1:
            xt1 = Hilbert_Resampling(xt1star, w, size, t)
        if t==T-1:
            return xt1star, w, np.linalg.norm(np.transpose(xt1star)@w)**2
            # add variance of each step
