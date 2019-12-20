from jax import grad, vmap, jit

import jax.scipy.stats as jsps
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as jsps

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import numpy as np

from gmm import (sample_gmm, gmm_gld, gmm_pdf_vectorized,
                 log_gmm_pdf_vectorized, log_lik_gmm)

from hmc import hmc_sampler


def get_plotfunc(true_samples):
    def plotfunc(particles):
        fig, ax = plt.subplots(1,1)

        kde1 = sns.kdeplot(true_samples.flatten(), ax=ax, label='Sampler')
        kde2 = sns.kdeplot(particles.flatten(), ax=ax, label='HMC')

        ax.legend()

        plt.savefig('./plots/hmc/1d_multimodal_medium_equal_other_side.png')
        plt.close()

    return plotfunc


if __name__ == '__main__':
    num_samples = 100

    weights = [1./5, 4./5]
    mus = [np.array([-4]),
           np.array([4])]

    sigmas = [np.eye(1),
              np.eye(1)]

    true_samples = sample_gmm(num_samples, weights, mus, sigmas)

    plotfunc = get_plotfunc(true_samples)

    log_pdf = log_gmm_pdf_vectorized(weights, mus, sigmas)

    target_energy = lambda x: -log_pdf(x)

    gld = gmm_gld(weights, mus, sigmas)

    target_grad = lambda x: -gld(x)

    #define other parameters
    params = {'u_energy': target_energy,
        'u_grad': target_grad,
        'step_size':1.5e-1, 
        'leapfrog_steps':20, 
        'total_samples':500, 
        'burn_in':.2, 
        'thinning_factor':2,
        'position_init': np.asarray([0.]).reshape(1,-1),
            'm':3}

    particles = hmc_sampler(**params)
    
    plotfunc(particles)