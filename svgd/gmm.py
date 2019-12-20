import random
import jax.numpy as jnp
from jax.scipy import stats as jsps
from jax import grad, vmap, jit
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp



def mvn_pdf(x, mu, sigma):
    k = len(mu)
    

    term1 = (2*jnp.pi)**(-k/2)
    term2 = 1./jnp.sqrt(jnp.linalg.det(sigma))
    term3 = jnp.exp(-1./2 * (x - mu).T @ jnp.linalg.inv(sigma) @ (x - mu))

    return term1 * term2 * term3


def mvn_pdf_vectorized(x, mu, sigma):
    """
    assumes x is a num_samples x num_dimensions array.
    performs the density computations in a vectorized manner
    """
    k = len(mu)

    demeaned_x = x - mu

    first_prod = jnp.linalg.inv(sigma) @ demeaned_x.T

    second_prod = demeaned_x * first_prod.T

    reduction = np.sum(second_prod, axis=1).flatten()

    term1 = (2*jnp.pi)**(-k/2)
    term2 = 1./jnp.sqrt(jnp.linalg.det(sigma))
    term3 = jnp.exp(-1./2 * reduction)

    return term1 * term2 * term3 

"""
For the below functions:
weights: the weight on each mode. Sum cannot exceed 1
mus: list of means
sigmas: list of covariance matrices
* weights, mus and sigmas must have same length
"""
# Outputs function for the gmm PDF
def gmm_pdf(weights, mus, sigmas):
    def p(x):

        pdfval = 0

        for weight, mu, sigma in zip(weights, mus, sigmas):
            pdfval = pdfval + weight * mvn_pdf(x, mu, sigma)

        return pdfval

    return p

# Vectorized version of gmm pdf
def gmm_pdf_vectorized(weights, mus, sigmas):
    def p(x):

        pdfval = 0

        for weight, mu, sigma in zip(weights, mus, sigmas):
            pdfval = pdfval + weight * mvn_pdf_vectorized(x, mu, sigma)

        return pdfval

    return p

# Finite differences function to check the analytic GLD
def finite_diff(g, h, weights, mus, sigmas):
    def f(x):
        return (g(x + h, weights, mus, sigmas) - g(x, weights, mus, sigmas))/h

    return f

# Outputs function for log of the PDF
def log_gmm_pdf(weights, mus, sigmas):
    p = gmm_pdf(weights, mus, sigmas)

    def logp(x):
        return jnp.log(p(x))

    return logp

# Vectorized version of log gmm pdf
def log_gmm_pdf_vectorized(weights, mus, sigmas):
    p = gmm_pdf_vectorized(weights, mus, sigmas)

    def logp(x):
        return jnp.log(p(x))

    return logp

# Outputs log likelihood of gmm
def log_lik_gmm(weights, mus, sigmas ):

    p = gmm_pdf_vectorized(weights, mus, sigmas)

    def log_lik(x):
        return jnp.sum(jnp.log(p(x)))

    return log_lik

# Produces analytic gradient log density for gmm
def gmm_gld(weights, mus, sigmas):

    logp = log_gmm_pdf(weights, mus, sigmas)

    return jit(vmap(grad(logp)))

# Produces samples from gmm
def sample_gmm(num_samples, weights, mus, sigmas):

    num_pdfs = len(weights)

    result = []
    for i in range(num_samples):
        # choose distribution
        idx = np.random.choice(range(num_pdfs), p=weights)

        mu = mus[idx]
        sigma = sigmas[idx]

        # sample from distribution
        sample = np.random.multivariate_normal(mu, sigma)

        result.append(sample)

    return np.array(result)


