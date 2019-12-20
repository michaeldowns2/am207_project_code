import time
from jax import grad, vmap, jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as jsps

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy as sp
import scipy.spatial as spsp

from scipy.stats import multivariate_normal

from gmm import log_gmm_pdf_vectorized
import matplotlib.pyplot as plt

def hmc_sampler(**kwargs):
    """ Samples from a target distribution converted to energy function u_energy
    Uses Euclidean-Gaussian Kinetic Energy and Leap-frog integrator
    """
    ## Setup
    t0 = time.time()
    #read in all arguments
    u_energy = kwargs['u_energy']
    step_size = kwargs['step_size']
    leapfrog_steps = kwargs['leapfrog_steps']
    total_samples = kwargs['total_samples']
    burn_in = kwargs['burn_in']
    thinning_factor = kwargs['thinning_factor']
    m = kwargs['m']
    #init a vector to hold the samples
    samples = []
    #start with specified initial position
    q_curr = kwargs['position_init']
    #define the dimension of p
    d = len(q_curr.flatten())
    #define the gradient of U
    ugrad = kwargs['u_grad']
    
    ## Params for kinetic energy
    #define mean
    mu = np.repeat(0, d)
    #define variance
    sig = np.identity(d) * m
    
    #set current u_energy
    ue_curr = u_energy(q_curr)
    
    #keep track of acceptance probability
    ap = 0
    
    # Repeat total_samples times
    for sam in range(total_samples):
        if sam % 100 == 0:
            print(f'At iteration {sam}')
        ## Step A: kick-off
        #sample random momentum
        p_curr = np.random.multivariate_normal(mu, sig)
    
        ## Step B: simulate movement
        #repeat for leapfrog_steps-1 times
        p_step = np.copy(p_curr)
        q_step = np.copy(q_curr)
        for step in range(leapfrog_steps):
            #half-step update for momentum
            p_step = p_step - step_size/2*ugrad(q_step)
            #full-step update for potential
            q_step = q_step + step_size/m*p_step
            #half-step update for momentum
            p_step = p_step - step_size/2*ugrad(q_step)
            
        ## Step C: Reverse momentum
        p_step = -p_step
        
        ## Step D: Correction for simulation error
        #compute total energy at current and step
        h_curr = ue_curr + 0.5 / m * np.linalg.norm(p_curr) ** 2
        h_step = u_energy(q_step) + 0.5 / m * np.linalg.norm(p_step) ** 2
        #generate alpha
        alpha = min(1, (np.exp(h_curr - h_step)))
        #sample from uniform
        u = np.random.uniform()
        #MH step
        if u <= alpha:
            #accept
            q_curr = q_step
            ue_curr = u_energy(q_curr)
            ap += 1
        #append whatever is current
        samples.extend(q_curr)
    
    #convert samples to numpy array
    samples = np.asarray(samples)
    #print acceptance rate
    print(f'The acceptance rate is {ap/total_samples:0.3f}')
    # Burning and thinning
    t1 = time.time()
    print(f'HMC Sampler took {t1-t0} seconds.')
    return samples[round(burn_in*len(samples))::thinning_factor, :]