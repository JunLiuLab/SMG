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
from os import path
import csv


a, b, d, e = sys.argv[1:]
n_dim = int(a) # number of dimensions
n_particles = int(b) # number of particles
rho = float(d)# rho
ess_ratio = float(e) # ess ratio




def save_result(filenames, to_save):
    for filename in filenames:
        try:
            if(not path.exists(filename)):
                with open(filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    head = ['T', 'n', 'rho', 'ess_ratio', 'proposal', 'mean', 'median', 'sd']
                    writer.writerow(head)

            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                head = to_save
                writer.writerow(head)
            # with open(filename, 'a', newline='') as f:
            #     writer = csv.writer(f)
            #     head = [n_dim, n_particles, rho, ess_ratio, 'random_walk', np.mean(res_log_normal_rw), np.median(res_log_normal_rw), np.std(res_log_normal_rw)]
            #     writer.writerow(head)
        except:
            pass

def Oracle(T = 10):
    x_prop = [[np.array([0,0])]]
    for t in range(1, T):
        new = []
        for xi in x_prop:
            up = np.array(xi[t-1]) + np.array([0,1])
            down = np.array(xi[t-1]) + np.array([0,-1])
            left = np.array(xi[t-1]) + np.array([-1,0])
            right = np.array(xi[t-1]) + np.array([1,0])
            legal = []
            for possible_xt in [up, down, left, right]:
                if tuple(possible_xt) not in [tuple(xx) for xx in xi]:
                    legal.append(xi + [possible_xt])
            new = new + legal
        num = len(new)
        save_result([filename_w, filename_l], [t+1, 0, 0, 0, 'N/A', num, num, 0])
        x_prop = new
    return len(x_prop)


filename_w = '/n/jun_liu_lab/wenshuowang/saw_oracle.csv'
filename_l = '/n/jun_liu_lab/yichaoli/saw_oracle.csv'

Oracle(n_dim)

