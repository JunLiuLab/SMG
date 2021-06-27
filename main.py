


a, b, c, d = sys.argv[1:]
n_dim = int(a) # number of dimensions
n_particles = int(b) # number of particles
n_multiple_des = int(c) # number of decsendants

%run -i high_dimensional.py

import matplotlib.pyplot as plt

def log_target_f(t, x):
    # defining the log density of the target
    return(math.log(multivariate_normal.pdf(x[0:(t+1)],3*np.ones(t+1),2*np.eye(t+1))+multivariate_normal.pdf(x[0:(t+1)],-3*np.ones(t+1),2*np.eye(t+1))))
sd = 3 # standard deviation of trial distribution

if d == 'exponential':
    rhos_used = np.array([(t+1)/T for t in range(T)])
else:
    rhos_used = np.array([float(d)]*T)

np.random.seed(2020)
# printing steps
Samples_iid, weights_iid, MSE_iid = Sampling(T = n_dim, size = n_particles, multiple_des = n_multiple_des, sd = sd, prop = 'i.i.d.', resample = Hilbert_Resampling, print_step = True)
Samples_SMG, weights_SMG, MSE_SMG = Sampling(T = n_dim, size = n_particles, multiple_des = n_multiple_des, sd = sd, prop = 'SMG', resample = Hilbert_Resampling, print_step = False)
Samples_more_particles, weights_more_particles, MSE_more_particles = Sampling(T = n_dim, size = n_multiple_des*n_particles, multiple_des = 1, sd = sd, prop = 'SMG', resample = Hilbert_Resampling, print_step = False)

# print MSEs
print(a, b, c, d)
print('i.i.d.:', MSE_iid, '\nSMG:', MSE_SMG, '\n'+str(n_multiple_des)+' times particles:', MSE_more_particles)


plt.style.use(u'default')
plt.rcParams['figure.figsize'] = (15,5)
# plt.plot(list(range(nT)), list(np.sum((np.array(res_HC[nt]) - np.array(est_ora[nt]))**2) for nt in range(nT)),color='black')
# plt.plot(list(range(nT)), list(np.sum((np.array(res_OT[nt]) - np.array(est_ora[nt]))**2) for nt in range(nT)),color='blue')


# fig = plt.figure(figsize = (15,5))
ax1 = fig.add_subplot(1,3,1, adjustable='box', aspect=1, xlim = (-7,7), ylim = (-7,7))
ax2 = fig.add_subplot(1,3,2, adjustable='box', aspect=1, xlim = (-7,7), ylim = (-7,7))
ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=1, xlim = (-7,7), ylim = (-7,7))

samples_iid = Hilbert_Resampling(Samples_iid, weights_iid, n_particles, n_dim-1)
ax1.scatter(samples_iid[:,0],samples_iid[:,1])
ax1.set_title(str(n_particles)+" particles, "+str(n_multiple_des)+' descendants i.i.d.')

samples_SMG = Hilbert_Resampling(Samples_SMG, weights_SMG, n_particles, n_dim-1)
ax2.scatter(samples_SMG[:,0],samples_SMG[:,1])
ax2.set_title(str(n_particles)+" particles, "+str(n_multiple_des)+' descendants SMG')

samples_more = Hilbert_Resampling(Samples_more_particles, weights_more_particles, n_particles*n_multiple_des, n_dim-1)
ax3.scatter(samples_more[:,0],samples_more[:,1])
ax3.set_title(str(n_multiple_des*n_particles)+" particles, 1 descendant")
# plt.show()
plt.savefig('SMG'+ a +'_'+ b + '_'+ c + '_' + d + '.png',bbox_inches='tight')


# plotting the last two dimensions; results are random because we need to resample to remove the weights
plt.style.use(u'default')
plt.rcParams['figure.figsize'] = (15,5)
# plt.plot(list(range(nT)), list(np.sum((np.array(res_HC[nt]) - np.array(est_ora[nt]))**2) for nt in range(nT)),color='black')
# plt.plot(list(range(nT)), list(np.sum((np.array(res_OT[nt]) - np.array(est_ora[nt]))**2) for nt in range(nT)),color='blue')


# fig = plt.figure(figsize = (15,5))
ax1 = fig.add_subplot(1,3,1, adjustable='box', aspect=1, xlim = (-7,7), ylim = (-7,7))
ax2 = fig.add_subplot(1,3,2, adjustable='box', aspect=1, xlim = (-7,7), ylim = (-7,7))
ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=1, xlim = (-7,7), ylim = (-7,7))

samples_iid = Hilbert_Resampling(Samples_iid, weights_iid, n_particles, n_dim-1)
ax1.scatter(samples_iid[:,n_dim-2],samples_iid[:,n_dim-1])
ax1.set_title(str(n_particles)+" particles, "+str(n_multiple_des)+' descendants i.i.d.')

samples_SMG = Hilbert_Resampling(Samples_SMG, weights_SMG, n_particles, n_dim-1)
ax2.scatter(samples_SMG[:,n_dim-2],samples_SMG[:,n_dim-1])
ax2.set_title(str(n_particles)+" particles, "+str(n_multiple_des)+' descendants SMG')

samples_more = Hilbert_Resampling(Samples_more_particles, weights_more_particles, n_particles*n_multiple_des, n_dim-1)
ax3.scatter(samples_more[:,n_dim-2],samples_more[:,n_dim-1])
ax3.set_title(str(n_multiple_des*n_particles)+" particles, 1 descendant")
plt.savefig('SMG'+ a +'_'+ b + '_'+ c + '_' + d + '.png',bbox_inches='tight')













