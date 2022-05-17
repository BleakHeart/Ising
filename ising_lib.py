import numpy as np
from numpy.random import rand
import time
from numba import jit, prange
from multiprocessing import Pool, cpu_count
import pandas as pd 
from tqdm import tqdm

def hot_config(L):
    return np.random.choice([-1, 1], size=(L, L))

def cold_config(L):
    return np.ones((L, L))

@jit
def E_dimensionless(config, L, J, H):
    total_energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i, j]
            nb = config[(i + 1) % L, j] + config[i, (j + 1) % L] + config[(i - 1) % L, j] + config[i, (j - 1) % L]
            total_energy += -J * nb * S - H * nb
    return total_energy / 2.


@jit
def MC_step(config, beta, J, H):
    '''
    Monte Carlo move using Metropolis algorithm
    '''
    L = len(config)
    for i in range(L*L):
        a = np.random.randint(0, L)  # looping over i & j therefore use a & b
        b = np.random.randint(0, L)
        sigma = config[a, b]
        neighbors = config[(a + 1) % L, b] + config[a, (b + 1) % L] + config[(a - 1) % L, b] + config[
            a, (b - 1) % L]
        del_E = 2 * J * sigma * neighbors + 2 * H * sigma
        
        if (del_E < 0) or (rand() < np.exp(-del_E * beta)):
            config[a, b] = -sigma


def phase_transition(T):
    # L is the length of the lattice

    # number of temperature points
    eqSteps = 50000
    #eqSteps = 5000
    mcSteps = 1000
    err_runs = 100
    coeff = np.log(1 + np.sqrt(2))

    T_c = 2 / np.log(1 + np.sqrt(2))

    # the number of MC sweeps for equilibrium should be at least equal to the number of MC sweeps for equilibrium

    # initialization of all variables
   
    E, E_std = 0, 0
    M, M_std, M_th = 0, 0, 0
    C, C_std, C_th = 0, 0, 0
    X, X_std = 0, 0

    # initialize total energy and mag
    beta = 1. / T
    # evolve the system to equilibrium
    for i in range(eqSteps):
        MC_step(config, beta)
    # list of ten macroscopic properties
    Ez = []
    Cz = []
    Mz = []
    Xz = []

    for j in range(err_runs):
        E = np.zeros(mcSteps)
        M = np.zeros(mcSteps)
        for i in range(mcSteps):
            MC_step(config, beta)
            E[i] = E_dimensionless(config, L)  # calculate the energy at time stamp
            M[i] = abs(np.mean(config))  # calculate the abs total mag. at time stamp


        # calculate macroscopic properties (divide by # sites) and append
        Energy = E.mean() / L ** 2
        SpecificHeat = beta ** 2 * E.var() / L**2
        Magnetization = M.mean()
        Susceptibility = beta * M.var() * (L ** 2)

        Ez.append(Energy)
        Cz.append(SpecificHeat)
        Mz.append(Magnetization)
        Xz.append(Susceptibility)

    E = np.mean(np.array(Ez))
    E_std = np.std(np.array(Ez))

    M = np.mean(np.array(Mz))
    M_std = np.std(np.array(Mz))

    C = np.mean(np.array(Cz))
    C_std = np.std(np.array(Cz))

    X = np.mean(np.array(Xz))
    X_std = np.std(np.array(Xz))
    
    if T - T_c >= 0:
        C_th = 0
        M_th = 0
    else:
        M_th = np.power(1 - np.power(np.sinh(2 * beta), -4), 1 / 8)
        C_th = (2.0 / np.pi) * (coeff ** 2) * (
                -np.log(1 - T / T_c) + np.log(1.0 / coeff) - (1 + np.pi / 4))
    
    return np.array([T, E, E_std, M, M_std, M_th, C, C_std, C_th, X, X_std])


@jit
def MC_step_single(config, beta, J, H):
    L = config.shape[0]
    '''
    Monte Carlo move using Metropolis algorithm
    '''
    a = np.random.randint(0, L)  # looping over i & j therefore use a & b
    b = np.random.randint(0, L)
    sigma = config[a, b]
    neighbors = config[(a + 1) % L, b] + config[a, (b + 1) % L] + config[(a - 1) % L, b] + config[
        a, (b - 1) % L]
    del_E = -2 * J * (-sigma) * neighbors - H * (-sigma)

    if (del_E < 0) or (rand() < np.exp(-del_E * beta)):
        sigma *= -1

    config[a, b] = sigma


def corr(T):
    Mcsteps = 4*10**6
    Mz = np.zeros(Mcsteps)

    for i in range(Mcsteps):
        MC_step_single(config, 1./T)
        Mz[i] = np.mean(config)
    return Mz


def entr(T):
    Mcsteps = 15*10**5
    
    p = {1: np.zeros(100), -1: np.zeros(100)}
    
    if T<2.0:
        H = 0.5
    else: 
        H = 0
    
    for r in range(100):
        config = cold_config(L)
        for i in range(Mcsteps):
            MC_step_single(config, 1./T)

        for j in [1,-1]:
            p[j][r] = (config == j).sum()
    
    e = 0.
    for j in [1,-1]:
            e -= p[j].mean() * np.log2(p[j].mean()/(L * L))/(L*L) 
    
    return e