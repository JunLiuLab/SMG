# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:28:33 2020

@author: lucas_lyc
"""


import pandas as pd
import numpy as np
import math
from numpy import random
from joblib import Parallel, delayed
import scipy
import sys
import csv
import ot
from scipy.stats import norm, multivariate_normal, uniform
import timeit
import json
from sklearn.decomposition import PCA
from os import path
from hilbertcurve.hilbertcurve import HilbertCurve

sys.path.append('/particle-filter/')
#from HDSV_generate_data import GeneratingData
#from RS import Residual_Sampling

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



def GeneratingData(rs = 123, num = 500, alpha = 0.7, mu = None, Sigma = None, beta = 1, p = 2):
    random.seed(rs)
    if not mu is None:
        if not Sigma is None:
            if not len(mu)==len(Sigma):
                return -1
            p = len(mu)
        else:
            p = len(mu)
            Sigma = np.eye(p,dtype=int)
    else:
        if not Sigma is None:
            p = len(Sigma)
            mu = np.zeros(p)
        else:
            Sigma = np.eye(p,dtype=int)
            mu = np.zeros(p)
    xt1 = random.multivariate_normal(mu,Sigma)
    # w = random.multivariate_normal(np.ones(p),np.eye(p,dtype=int))
    yt1 = np.zeros(p)
    for j in range(p):
        yt1[j] = beta * math.exp(xt1[j]/2)*random.normal()
    X = [xt1]
    Y = [yt1]
    for i in range(num -1):
        xt = random.multivariate_normal(mu,Sigma) + alpha* xt1
        yt = np.zeros(p)
        for j in range(p):
            yt[j] = beta * math.exp(xt[j]/2)*random.normal()
        X.append(xt)
        Y.append(yt)
        xt1 = xt.copy()
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def Multinomial_Resampling(particles, weights, size):
    indices = [random.choice(range(len(particles)), p = weights) for _ in range(size)]
    res = np.array([particles[i] for i in indices])
    return res

def Residual_Matrix(w, M):
    N = len(w)
    res = np.zeros([M,N])
    adjusted_weights = M*w
    k = 0
    for j in range(N):
        while adjusted_weights[j]>=1:
            res[k,j] = 1
            k = k + 1
            adjusted_weights[j] = adjusted_weights[j] - 1
    if k==M:
    	return res
    adjusted_weights = adjusted_weights * (1/(M-k))
    for j in range(k,M):
        res[j,:] = adjusted_weights
    return res

def Residual_Resampling(particles, weights, size):
    weight_matrix = Residual_Matrix(weights, size)
    indices = [random.choice(range(len(particles)), p = w) for w in weight_matrix]
    res = np.array([particles[i] for i in indices])
    return res

def OT_Resampling(particles, weights, size, reg = .1):
    M = np.matrix([[np.linalg.norm(xi-xj)**2 for xi in particles] for xj in particles])
    row = np.array([1/size]*size)
    weight_matrix = ot.sinkhorn(row, weights, M, reg = reg) 
    weight_matrix = weight_matrix*size
    try:
        indices = [random.choice(range(len(particles)), p = w) for w in weight_matrix]
    except:
        try:
            weight_matrix = np.apply_along_axis(lambda x: x - np.min(np.min(x), 0) , axis = 1, arr = weight_matrix)
            weight_matrix = np.apply_along_axis(lambda x: x/np.sum(x) , axis = 1, arr = weight_matrix)
            indices = [random.choice(range(len(particles)), p = w) for w in weight_matrix]
        except:
            indices = [random.choice(range(len(particles)), p =weights) for _ in range(size)]
    res = np.array([particles[i] for i in indices])
    return res


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

def Stratified_Resampling(particles, weights, size):
    weight_matrix = Stratified_Matrix(weights, size)
    indices = [random.choice(range(len(particles)), p = w) for w in weight_matrix]
    res = np.array([particles[i] for i in indices])
    return res  

def Hilbert_Resampling(particles, weights, size):
    particles = np.array(particles)
    dim = particles.shape[1]
    pmax = [max(particles[:,k])+0.1 for k in range(dim)]
    pmin = [min(particles[:,k])-0.1 for k in range(dim)]
    unified_particles = np.array([[(par[k]-pmin[k])/(pmax[k]-pmin[k]) for k in range(dim)] for par in particles])
    hilbert_mapping = [Hilbert_Mapping(up, dim=dim) for up in unified_particles]
    Weighted_Sample = pd.concat([pd.DataFrame(particles),pd.DataFrame({"weight": weights, 'map': hilbert_mapping})],axis=1)
    Weighted_Sample = Weighted_Sample.sort_values(by = ['map'], ascending = True)
    Weighted_Sample.index = range(Weighted_Sample.shape[0])
    w = Weighted_Sample['weight']
    weight_matrix = Stratified_Matrix(w, M = size)
    
    indices = [random.choice(range(len(particles)), p = w) for w in weight_matrix]
    res = np.array(Weighted_Sample.iloc[indices, list(range(dim))])
    return res

def Multiple_Descendent_Proposal(particles, y, alpha = 0.7, Sigma = Sigma, beta = beta, multiple_des = 4):
    x_prop = []
    weight = []
    p = particles.shape[1]
    for par in particles:
        for k in range(multiple_des):
            x_prop.append(random.multivariate_normal(np.zeros(p), Sigma) + alpha* par)
            weight.append(0)
            for jj in range(p): # calculate log-likelihood
                sig = beta*math.exp(x_prop[-1][jj]/2)
                weight[-1] = weight[-1] + norm.logpdf(y[jj], 0, sig)
    x_prop = np.array(x_prop)
    weight = np.array(weight)
    weight = weight - np.max(weight)
    weight = np.exp(weight)
    weight = weight/np.sum(weight)
    #print('dp', np.sum(weight**2))
    return x_prop, weight

def Hilbert_Stratified_Proposal(particles, y, alpha = 0.7, Sigma = Sigma, beta = beta, multiple_des = 4):
    x_prop = []
    weight = []
    p = particles.shape[1]
    for par in particles:
        hh = [(uniform.rvs() + md)/multiple_des for md in range(multiple_des)]
        vv = [Hilbert_Mapping_Inverse(h, dim = p) for h in hh]
        xx = [norm.ppf(v) for v in vv]
        for k, x_star in enumerate(xx):
            x_prop.append(scipy.linalg.sqrtm(Sigma)@x_star + alpha*par)
            weight.append(0)
            for jj in range(p): # calculate log-likelihood
                sig = beta*math.exp(x_prop[-1][jj]/2)
                weight[-1] = weight[-1] + norm.logpdf(y[jj], 0, sig)
    x_prop = np.array(x_prop)
    weight = np.array(weight)
    weight = weight - np.max(weight)
    weight = np.exp(weight)
    weight = weight/np.sum(weight)
    #print('sp', np.sum(weight**2))
    return x_prop, weight

def Multi_SV(obs, size = 100, mu =None, p = 2, alpha = 0.7, Sigma = None, beta = beta, multiple_des = 4, 
                  prop = Multiple_Descendent_Proposal, resample = Multinomial_Resampling):
    if mu is None:
        mu = np.zeros(p)
    if Sigma is None:
        Sigma = np.eye(p, dtype=int)
    w = np.zeros(size)
    xt1= random.multivariate_normal(mu,Sigma,size = size)
    for i in range(size):
        for j in range(p):
            # print(i)
            sig = beta*math.exp(xt1[i,j]/2)
            w[i] += norm.logpdf(obs[0,j], 0, sig)
    w = [x-max(w) for x in w]
    w = [math.exp(x) for x in w]
    w = [x/sum(w) for x in w]
    w = np.array(w)
    vol = [np.transpose(xt1)@w]
    xt1 = resample(xt1, w, size)
        
    for i in range(1, len(obs)):
        xt1star, w = prop(xt1, obs[i], multiple_des = multiple_des)
        vol.append(np.transpose(xt1star)@w)
        xt1 = resample(xt1star, w, size)
        
    vol = np.array(vol)
    return vol