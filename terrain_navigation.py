import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib import cm
from scipy.stats import norm, uniform
from hilbertcurve.hilbertcurve import HilbertCurve
from scipy import interpolate
import sys
import timeit
import csv

data = pd.read_csv('colorado.dat', sep = ' ', header = None)
data = np.array(data)[:161,:]
data_lat = data[0, 1:]
data_lon = data[1:, 0]
data_elev = data[1:, 1:]

delta = 40
q = 200
k = 1/2
ntimes = 200
n_particles = 100
f = interpolate.interp2d(x = data_lat, y = data_lon, z= data_elev)

def Multinomial_Resampling(particles, weights, size):
    indices = random.choice(range(len(particles)), size = size, p = weights)
    res = np.array([particles[i] for i in indices])
    return res


def Hilbert_Mapping(x, p = 8):
    hilbert_curve = HilbertCurve(p, 2)
    aa = [int(xx) for xx in x*2**p]
    h = hilbert_curve.distance_from_coordinates(aa)
    h = np.array(h/(2**(p*2)))
    return h

def Stratified_Resampling(ww, M):
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

def Hilbert_Resampling(particles, weights, size):
    x_max, x_min = max(particles[:,0]) + 100, min(particles[:,0]) - 100
    y_max, y_min = max(particles[:,1]) + 100, min(particles[:,1]) - 100
    unified_particles = np.array([[(par[0] - x_min)/(x_max - x_min), (par[1] - y_min)/(y_max - y_min)] for par in particles])
    hilbert_mapping = [Hilbert_Mapping(up) for up in unified_particles]
    Weighted_Sample = pd.DataFrame({'sample_x': particles[:,0], 'sample_y': particles[:,1], 
                                    "weight": weights, 'map': hilbert_mapping})
    Weighted_Sample = Weighted_Sample.sort_values(by = ['map'], ascending = True)
    Weighted_Sample.index = range(Weighted_Sample.shape[0])
    w = Weighted_Sample['weight']
    weight_matrix = Stratified_Resampling(w, M = size)
    indices = [random.choice(range(len(particles)), p = w) for w in weight_matrix]
    res = np.array(Weighted_Sample.iloc[indices, [0, 1]])
    return res

def Optimal_Transport_Resampling(particles, weights, size, reg = 1e4):
    M = np.matrix([[np.linalg.norm(xi-xj)**2 for xi in particles] for xj in particles])
    row = np.array([1/size]*size)
    weight_matrix = ot.sinkhorn(row, weights, M, reg = reg) 
    weight_matrix = weight_matrix*size
    indices = [random.choice(range(len(particles)), p = w) for w in weight_matrix]
    res = np.array([particles[i] for i in indices])
    return res


def Multiple_Descendent_Proposal(particles, y, drift, q, multiple_des = 4):
    track_prop = []
    weight = []
    for par in particles:
        for k in range(multiple_des):
            xt, yt = par
            Rt = np.matrix([[-xt, yt], [-yt, -xt]])/np.sqrt((xt**2 + yt**2))
            Zt = random.multivariate_normal(np.zeros(2), q**2*np.matrix([[1, 0], [0, k**2]]))
            Et = np.array(Rt.T)@Zt
            xnew = xt + drift[0] + Et[0]
            ynew = yt + drift[1]+ Et[1]
            track_prop.append([xnew, ynew])
            weight.append(norm.logpdf(f(xnew, ynew)[0], y, scale = delta))
    track_prop = np.array(track_prop)
    weight = np.array(weight)
    weight = weight-max(weight)
    weight = np.exp(weight)
    weight = weight/np.sum(weight)
    #print('dp', np.sum(weight**2))
    return track_prop, weight

def Hilbert_Mapping_Inverse(h, p = 8, dim = 10):
    hilbert_curve = HilbertCurve(p, dim)
    aa = hilbert_curve.coordinates_from_distance(int(h*2**(p*dim)))
    aa = np.array(aa) + 0.5
    aa = aa/(2**p)
    return aa

def Hilbert_Stratified_Proposal(particles, y, drift, q, multiple_des = 4):
    track_prop = []
    weight = []
    for par in particles:
        hh = [(uniform.rvs() + md)/multiple_des for md in range(multiple_des)]
        vv = [Hilbert_Mapping_Inverse(h, dim = 2) for h in hh]
        xx = [norm.ppf(v) for v in vv]
        for xstar in xx:
            xt, yt = par
            Rt = np.matrix([[-xt, yt], [-yt, -xt]])/np.sqrt((xt**2 + yt**2))
            Et = np.array(Rt.T)@np.diag([q, q*k])@xstar
            xnew = xt + drift[0] + Et[0]
            ynew = yt + drift[1] + Et[1]
            track_prop.append([xnew, ynew])
            weight.append(norm.logpdf(f(xnew, ynew)[0], y, scale = delta))
    track_prop = np.array(track_prop)
    weight = np.array(weight)
    weight = weight-max(weight)
    weight = np.exp(weight)
    weight = weight/np.sum(weight)
    #print('dp', np.sum(weight**2))
    return track_prop, weight



def Tracking(zt, q = 200,  n_particles = n_particles, method = 'multinomial', multiple_des = 4, prop = Multiple_Descendent_Proposal):
    x0 = 0
    y0 = 30000
    f = interpolate.interp2d(x = data_lat, y = data_lon, z= data_elev)
    z0 = f(x0, y0)[0]
    x_particles = np.repeat(x0, n_particles)
    y_particles = np.repeat(y0, n_particles)
    track_estimated_x = [x0]
    track_estimated_y = [y0]
    weight = np.repeat(1/n_particles, n_particles)
    for i in range(ntimes):
        particles = list(zip(x_particles, y_particles))
        drift = [drift_x[i], drift_y[i]]
        track_prop, weight = prop(particles, zt[i], drift, q, multiple_des = multiple_des)
        #resampling
        if method == 'multinomial':
            new = Multinomial_Resampling(track_prop, weight, n_particles)
        elif method == 'ot':
            new = Optimal_Transport_Resampling(track_prop, weight, n_particles)
        elif method == 'residual':
            new = Residual_Resampling(track_prop, weight, n_particles)
        elif method == 'ordered_stratified':
            new = Hilbert_Resampling(track_prop, weight, n_particles)
        else:
            return None
        xnew = new[:, 0]
        ynew = new[:, 1]
        weight = np.repeat(1/n_particles, n_particles)
        #estimate
        track_estimated_x.append(np.sum(weight*xnew))
        track_estimated_y.append(np.sum(weight*ynew))
        x_particles = xnew
        y_particles = ynew
    return track_estimated_x, track_estimated_y


#np.random.seed(1234)
#Set up true starting point

#Set up true drift
route_theta= np.arange(-np.pi/2, np.pi/2, np.pi/(ntimes + 1))
route_x = (15000*np.cos(route_theta))[::-1]
route_y = (15000*np.sin(route_theta))[::-1]
drift_x = np.diff(route_x)
drift_y = np.diff(route_y)
#drift_x = norm.rvs(size = 100, loc = 0, scale = 2000)
#drift_y = norm.rvs(size = 100, loc = 0, scale = 2000)


def compare(seed = 123321, q= 200):
    #Set up true track
    random.RandomState(seed)
    random.seed(seed)
    x0 = 0
    y0 = 30000
    f = interpolate.interp2d(x = data_lat, y = data_lon, z= data_elev)
    z0 = f(x0, y0)[0]
    
    xt = x0
    yt = y0
    track_x = [x0]
    track_y = [y0]
    zt = [f(x0, y0)[0]]
    for i in range(ntimes):
        Rt = np.matrix([[-xt, yt], [-yt, -xt]])/np.sqrt((xt**2 + yt**2))
        Zt = random.multivariate_normal(np.zeros(2), q**2*np.matrix([[1, 0], [0, k**2]]))
        Et = np.array(Rt.T)@Zt
        xt = xt + np.array(drift_x[i]) + Et[0]
        yt = yt + np.array(drift_y[i]) + Et[1]
        track_x.append(xt)
        track_y.append(yt)
        zt.append(f(xt, yt)[0]+ norm.rvs(scale = delta))
    
    start1 = timeit.default_timer()
    track_mul = Tracking(zt, q= q, method = 'multinomial')
    time_multi = timeit.default_timer() - start1
    start2 = timeit.default_timer()
    track_rs = Tracking(zt, q= q, method = 'residual')
    time_rs = timeit.default_timer() - start2
    start3 = timeit.default_timer()
    track_sr = Tracking(zt, q= q, method = 'ordered_stratified')
    time_hc = timeit.default_timer() - start3
    start4 = timeit.default_timer()
    track_mul2 = Tracking(zt, q= q, method = 'multinomial', prop = Hilbert_Stratified_Proposal)
    time_multi2 = timeit.default_timer() - start4
    start5 = timeit.default_timer()
    track_rs2 = Tracking(zt, q= q, method = 'residual', prop = Hilbert_Stratified_Proposal)
    time_rs2 = timeit.default_timer() - start5
    start6 = timeit.default_timer()
    track_sr2 = Tracking(zt, q= q, method = 'ordered_stratified', prop = Hilbert_Stratified_Proposal)
    time_hc2 = timeit.default_timer() - start6
    #track_oracle = Tracking(zt, q= q, n_particles = 10000, method = 'residual', multiple_des=1)
    
    
    
    XI = np.linspace(-6000, 34000, 100)
    YI = np.linspace(-6000, 34000, 100)
    ZI = f(XI, YI)
    plt.figure(figsize=(8,6))
    plt.pcolor(XI, YI, ZI, cmap=cm.Greys)
    plt.title('Terrain Navigation')
    plt.xlim(-6000, 34000)
    plt.ylim(-6000, 34000)
    plt.colorbar()
    plt.plot(track_x, track_y, color = 'red')
    plt.plot(track_mul[0], track_mul[1], color = 'blue')
    plt.plot(track_rs[0], track_rs[1], color = 'green')
    plt.plot(track_sr[0], track_sr[1], color = 'purple')
    plt.plot(track_mul2[0], track_mul2[1], '--', color = 'blue')
    plt.plot(track_rs2[0], track_rs2[1], '--', color = 'green')
    plt.plot(track_sr2[0], track_sr2[1], '--', color = 'purple')
    #plt.savefig('plots/'+str(sig)+'plot_real'+str(seed)+'.png')
    
    track_oracle = np.array([track_x, track_y])
    err_mul = np.sqrt(np.mean([((track_mul[0][i] - track_oracle[0][i])**2 + (track_mul[1][i] - track_oracle[1][i])**2) for i in range(ntimes)]))**2
    err_rs = np.sqrt(np.mean([((track_rs[0][i] - track_oracle[0][i])**2 + (track_rs[1][i] - track_oracle[1][i])**2) for i in range(ntimes)]))**2
    err_sr = np.sqrt(np.mean([((track_sr[0][i] - track_oracle[0][i])**2 + (track_sr[1][i] - track_oracle[1][i])**2) for i in range(ntimes)]))**2
    err_mul2 = np.sqrt(np.mean([((track_mul2[0][i] - track_oracle[0][i])**2 + (track_mul2[1][i] - track_oracle[1][i])**2) for i in range(ntimes)]))**2
    err_rs2 = np.sqrt(np.mean([((track_rs2[0][i] - track_oracle[0][i])**2 + (track_rs2[1][i] - track_oracle[1][i])**2) for i in range(ntimes)]))**2
    err_sr2 = np.sqrt(np.mean([((track_sr2[0][i] - track_oracle[0][i])**2 + (track_sr2[1][i] - track_oracle[1][i])**2) for i in range(ntimes)]))**2


    return err_mul, err_rs, err_sr, err_mul2, err_rs2, err_sr2, time_multi, time_rs, time_hc, time_multi2, time_rs2, time_hc2

def plot_route(seed = 123321, q= 200):
    random.seed(seed)
    #Set up true track
    x0 = 0
    y0 = 30000
    f = interpolate.interp2d(x = data_lat, y = data_lon, z= data_elev)
    z0 = f(x0, y0)[0]
    
    xt = x0
    yt = y0
    track_x = [x0]
    track_y = [y0]
    zt = [f(x0, y0)[0]]
    for i in range(ntimes):
        Rt = np.matrix([[-xt, yt], [-yt, -xt]])/np.sqrt((xt**2 + yt**2))
        Zt = random.multivariate_normal(np.zeros(2), q**2*np.matrix([[1, 0], [0, k**2]]))
        Et = np.array(Rt.T)@Zt
        xt = xt + np.array(drift_x[i]) + Et[0]
        yt = yt + np.array(drift_y[i]) + Et[1]
        track_x.append(xt)
        track_y.append(yt)
        zt.append(f(xt, yt)[0]+ norm.rvs(scale = delta))
    XI = np.linspace(-6000, 34000, 100)
    YI = np.linspace(-6000, 34000, 100)
    ZI = f(XI, YI)
    plt.figure(figsize=(8,6))
    plt.pcolor(XI, YI, ZI, cmap=cm.Greys)
    plt.title('Terrain Navigation')
    plt.xlim(-6000, 34000)
    plt.ylim(-6000, 34000)
    plt.colorbar()
    plt.plot(track_x, track_y, color = 'red')

    

