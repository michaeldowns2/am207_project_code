"""
Implementation of https://arxiv.org/pdf/1608.04471.pdf


different bandwidths


compare samples obtained via svgd with other methods

Investigate JAX numerical instability away from center of mass
JAX: numerical instability issues far from location of mass -- underflow?

"""
import logging
import random
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

#np.random.seed(207)

#random.seed(207)

class SVGD:
    def __init__(self,
                 gld=None,
                 mini_batch_size=-1,
                 debug=0):

        if mini_batch_size > -1:
            raise NotImplementedError

        if gld is None:
            raise ValueError("GLD not defined")

        self.gld = gld
        self.mini_batch_size = mini_batch_size
        self.debug = debug

        self.particles = []

    @staticmethod
    def rbf_param_function(X, dists):
        """
        X: Current data
        """

        return np.median(dists)**2 / np.log(len(X))

    @staticmethod
    def compute_rbf_kernel_matrix(particles, dists, h, num_particles):
        sq_dists = dists**2

        kernel_vals = np.exp(-1./h * sq_dists)

        K = np.zeros((num_particles, num_particles))

        K[np.triu_indices(num_particles, 1)] = kernel_vals

        K = K + K.T
        np.fill_diagonal(K, 1)

        return K

    @staticmethod
    def compute_rbf_kernel_deriv(particles, h, K_reshaped):
        broadcasted_subtraction = particles[:, np.newaxis] - particles

        broadcasted_subtraction = np.rot90(broadcasted_subtraction, axes=(2, 1))
        broadcasted_subtraction = np.rot90(broadcasted_subtraction, axes=(0, 2))

        broadcasted_multiplication = K_reshaped * broadcasted_subtraction

        partials_matrix = 2 * -1./h * broadcasted_multiplication

        return partials_matrix



    def do_svgd_iterations_optimized(self, init_particles,
                           num_iterations=1000,
                                     learning_rate=0.01,
                                     plotfunc=None,
                                     metrics=None,
                                     alpha=0.9,
                                     progress_freq=10):

        eps = 1e-6
        historical_updates = 0

        num_particles, dim = init_particles.shape
        current_particles = init_particles

        img_num = 0
        for l in range(num_iterations):
            if l == 0:
                if plotfunc is not None:
                    plotfunc(-1, current_particles)
            
            if self.debug: t = time.time()
            dists = spsp.distance.pdist(current_particles)
            if self.debug: print("Computed distances in {} seconds".format(time.time() - t))

            

            # get new kernel param
            if self.debug: t = time.time()
            h = SVGD.rbf_param_function(current_particles, dists)
            if self.debug: print("Computed kernel param in {} seconds".format(time.time() - t))


            # get gradient values in a vectorized fashion
            if self.debug: t = time.time()
            gld = np.array(self.gld(current_particles))
            if self.debug: print("Computed gld in {} seconds".format(time.time() - t))

            # rotate gld 90 degrees counterclockwise about axis
            # normal to yz plane
            gld_reshaped = gld.reshape(num_particles, dim, 1)
            gld_reshaped = np.rot90(gld_reshaped, axes=(0, 2))

            # get kernel matrix
            if self.debug: t = time.time()
            K = SVGD.compute_rbf_kernel_matrix(current_particles, dists,
                                               h, num_particles)

       
            if self.debug: print("Computed kernel matrix in {} seconds".format(time.time() - t))

            # rotate K 90 degrees counterclockwise about axis
            # normal to xy
            K_reshaped = K.reshape(num_particles, num_particles, 1)
            K_reshaped = np.rot90(K_reshaped, axes=(1, 2))


            # compute broadcasted product representing first term in SVGD updates
            if self.debug: t = time.time()
            term1 = K_reshaped * gld_reshaped
            if self.debug: print("Computed term1 in {} seconds".format(time.time() - t))

            term1 = term1.mean(axis=2)

            # get kernel matrix gradient
            if self.debug: t = time.time()
            dK = SVGD.compute_rbf_kernel_deriv(current_particles, h, K_reshaped)
            term2 = dK.mean(axis=2)
            if self.debug: print("Computed term2 in {} second".format(time.time() - t))


            updates = term1 + term2

            # do adagrad with momentum as per the author's implementation

            if l == 0:
                historical_updates = updates ** 2
            else:
                historical_updates = alpha * historical_updates + (1 - alpha) * (updates ** 2)

            adj_updates = np.divide(updates, eps + np.sqrt(historical_updates))

            current_particles = current_particles + (learning_rate) * adj_updates

            if self.debug: print()

            if l % round(num_iterations / progress_freq) == 0:
                print("Iteration {}, {}% finished".format(l, round(l/num_iterations * 100, 2)))

                if plotfunc is not None:
                    plotfunc(l, current_particles)

                if metrics is not None:
                    metrics(current_particles)



        return current_particles


   





    
