import numpy as np
import os
from scipy.stats import norm
import pickle

n = 500
n_sigs = [0.1, 0.3, 0.5, 0.7]


for n_sig in n_sigs:
    ce = []
    for i in range(0, n):
        c = np.random.uniform(low=0.0, high=1.0, size=1).astype(np.float32)
        noise = np.random.normal(0.0, n_sig, 1).astype(np.float32)
        e = -2*c*c + noise
        ce.append([c, e])
    ce = np.array(ce)
    ce = np.squeeze(ce, axis=2)
    data = {'X': ce}
    pickle.dump(data, open("sim/sim_cc_sig_"+str(n_sig)+".p", "wb"))


for n_sig in n_sigs:
    ce = []
    for i in range(0, n):
        c = np.random.uniform(low=0.0, high=1.0, size=1).astype(np.float32)
        noise = np.random.normal(0.0, n_sig, 1).astype(np.float32)
        e = c*np.log(c + 0.1) + noise
        ce.append([c, e])
    ce = np.array(ce)
    ce = np.squeeze(ce, axis=2)
    data = {'X': ce}
    pickle.dump(data, open("sim/sim_clog_sig_"+str(n_sig)+".p", "wb"))




