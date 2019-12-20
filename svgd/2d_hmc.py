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
    def plotfunc(l, particles):
        fig, ax = plt.subplots(1,1)

        sns.scatterplot(x=true_samples[:, 0].flatten(),
                           y=true_samples[:, 1].flatten(),
                           ax=ax,
                        color='blue',
                           label='Sampler')

        sns.scatterplot(x=particles[:, 0].flatten(),
                           y=particles[:, 1].flatten(),
                           ax=ax,
                        color='orange',
                           label='HMC')


        ax.legend()

        plt.savefig('./plots/hmc/bimodal_bivariate_gmm_medium_equal_correlation1.png')
        plt.close()

    return plotfunc

if __name__ == '__main__':
    num_iterations = 10000
    num_samples = 100

    weights = [0.5, 0.5]
    mus = [np.array([-4, 0]),
           np.array([4, 0])]
    corr1 = -0.33
    corr2 = 0.66
    sigmas = [np.array([
        [1, corr1],
        [corr1, 1]]),
        np.array([
        [1, corr2],
        [corr2, 1]
        ])]

    init_particles = sample_gmm(num_samples,
                                    [1.],
                                    [np.array([-10, 0])],
                                    [np.eye(2)])

    true_samples = sample_gmm(num_samples, weights, mus, sigmas)

    plotfunc = get_plotfunc(true_samples)

    log_pdf = log_gmm_pdf_vectorized(weights, mus, sigmas)

    target_energy = lambda x: -log_pdf(x)

    gld = gmm_gld(weights, mus, sigmas)

    target_grad = lambda x: -gld(x)

    #define other parameters
    params = {'u_energy': target_energy,
        'u_grad': target_grad,
        'step_size':1., 
        'leapfrog_steps':15, 
        'total_samples':300, 
        'burn_in':.2, 
        'thinning_factor':2,
        'position_init': np.asarray([0., 0.]).reshape(1,-1),
            'm':1}

    particles = hmc_sampler(**params)
    
    plotfunc(1, particles)