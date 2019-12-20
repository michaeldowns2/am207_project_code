from jax import grad, vmap, jit

import jax.scipy.stats as jsps
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as jsps

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import numpy as np

from gmm import sample_gmm, gmm_gld, gmm_pdf_vectorized, log_lik_gmm
from svgd import SVGD


def make_video(frames):
    fig = plt.figure()

    ani = animation.ArtistAnimation(fig, frames, interval=0.5)
    plt.show()
    ani.save('movie.mp4')
    plt.close()


def metrics(log_lik_func):
    def report_metrics(particles):
        print("Log Likelihood: {}".format(log_lik_func(particles)))

    return report_metrics

def get_plotfunc(true_samples):
    def plotfunc(l, particles):
        fig, ax = plt.subplots(1,1)

        kde1 = sns.kdeplot(true_samples.flatten(), ax=ax, label='Sampler')
        kde2 = sns.kdeplot(particles.flatten(), ax=ax, label='SVGD')

        ax.legend()

        plt.savefig('./plots/1d_multimodal_medium_equal_other_side/{}.png'.format(l))
        plt.close()

    return plotfunc


if __name__ == '__main__':
    num_iterations = 500
    num_samples = 100

    weights = [1./5, 4./5]
    mus = [np.array([-4]),
           np.array([4])]
    sigmas = [np.eye(1),
              np.eye(1)]

    toy_init_particles = sample_gmm(num_samples,
                                    [1.],
                                    [np.array([10])],
                                    [np.eye(1)])

    # a = -10
    # b = 10
    # toy_init_particles = np.random.rand(100).reshape(-1, 1)*(b - a) + a

    true_samples = sample_gmm(num_samples, weights, mus, sigmas)

    plotfunc = get_plotfunc(true_samples)

    log_lik_func = log_lik_gmm(weights, mus, sigmas)
    report_metrics = metrics(log_lik_func)

    toy_score = gmm_gld(weights, mus, sigmas)

    svgd = SVGD(gld=toy_score)

    particles = svgd.do_svgd_iterations_optimized(init_particles=toy_init_particles,
                                         num_iterations=num_iterations,
                                                  learning_rate=0.1,
                                                  plotfunc=plotfunc,
                                                 metrics=report_metrics)




