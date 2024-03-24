import numpy as np
import pandas as pd
from analogrb.gatefactory import UniformNNHGateFactory
from analogrb.channel import DepolarizingChannel
from analogrb.bosonic import OnsideIntNNHamiltonian, NonintNNHamiltonian
from analogrb.arb_protocol import simulate_data_acquisition, postprocess, aggregate

import subprocess
import numpy as np
import os
from scipy.special import binom
from copy import deepcopy
from typing import Union

import pathlib

from analogrb import basis


# number of nodes.
d = 3
# number of particles.
n = 2
# the y dimension of the lattice.
ydim = d
# Set if the system is interacting or not.
interacting_gates = False
# Build the Hamiltonian with which the system will be time evolved.
Hamiltonian = OnsideIntNNHamiltonian(d, n, ydim) if interacting_gates else NonintNNHamiltonian(d, n, ydim)
# The evolution time of the Hamiltonian
time = 1.
# When called, returns a random unitary, here a unitary from a uniformly drawn NN = nearest neighbour Hamiltonian. 
gatefactory = iter(UniformNNHGateFactory(Hamiltonian, time = time))
# The initial quantum state, here the |0x0| density matrix.
rho_init = np.zeros((Hamiltonian.fock_dim, Hamiltonian.fock_dim), dtype=np.complex128)
rho_init[0, 0] = 1.

# Set the error channel.
depol_param = 0.05
error_channel = DepolarizingChannel(depol_param)

# Set the parameters to run the arb protocol. ms are the depths of the quenches.
ms = [5, 10, 15]
# How many time the protocol is repeated.
naverage = 10

# Set the statistical analysis paramters. 
nbootstraps = 100
confidence = 95
start_fitting = 10

save = True
data_path = simulate_data_acquisition(d, n, interacting_gates, ms, gatefactory, rho_init, error_channel, naverage, save)

data = postprocess(data_path)
df = pd.DataFrame(data)

df_result, df_result_capped, names = aggregate(data_path, start_fitting, nbootstraps, confidence)

import matplotlib.pyplot as plt
from analogrb.save_load import extract_from_data
from analogrb.bootstrap import EXP_FUNC

COLORS = ['#469DD4', '#DB3932', '#638537',  '#853763',  '#D79B00'] # blue, red, green, purple, orange


for k, name in enumerate(names):
    ms, qs = extract_from_data(df, name, 'm', list)
    color = COLORS[k]
    plt.scatter(*extract_from_data(df, name, 'm'), alpha=0.1, color=color)
    ms = df_result.loc[name, 'm']
    plt.errorbar(ms, df_result.loc[name, 'avg'], yerr=df_result.loc[name, 'yerr'], fmt='x', capsize=2, color=color, label=f"irrep {name[2:]}")
    ms_finespaced = np.linspace(np.min(ms), np.max(ms), len(ms)*20)
    plt.plot(ms_finespaced, EXP_FUNC(ms_finespaced, *df_result.loc[name, 'popt']), '--', color = color)
    print(f"irrep {name[2:]} popt: A, p = {df_result.loc[name, 'popt']}")
plt.ylabel('q')
plt.xlabel('m')
plt.legend()
plt.show()
