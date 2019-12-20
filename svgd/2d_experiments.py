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

from svgd import SVGD



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
                           label='SVGD')


        ax.legend()

        plt.savefig('./plots/bimodal_bivariate_gmm_medium_equal_correlation1/{}.png'.format(l))
        plt.close()

    return plotfunc


def metrics(log_lik_func):
    def report_metrics(particles):
        df = pd.DataFrame(particles)
        corr = df.corr().iloc[0, 1]
        print("Log Likelihood: {} | Corr: {}".format(log_lik_func(particles), corr))

    return report_metrics


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

    init_particles = sample_gmm(5,
                                    [1.],
                                    [np.array([-10, 0])],
                                    [np.eye(2)])

    true_samples = sample_gmm(num_samples, weights, mus, sigmas)

    plotfunc = get_plotfunc(true_samples)

    gld = gmm_gld(weights, mus, sigmas)

    svgd = SVGD(gld=gld)

    log_lik_func = log_lik_gmm(weights, mus, sigmas)

    report_metrics = metrics(log_lik_func)

    particles = svgd.do_svgd_iterations_optimized(
        init_particles=init_particles,
        num_iterations=num_iterations,
        learning_rate=1e-3,
        plotfunc=plotfunc,
        metrics=report_metrics,
        progress_freq=100)


  
