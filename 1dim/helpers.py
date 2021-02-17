
# helpers.py

import os
import math
import numpy as np
import scipy.special as spec

import config as cf


## FUNCTIONS FOR COMPUTING M-ESTIMATORS (AND THE LIKE) ##

def rho_catnew(u):
    '''
    Parent derived from the influence function of
    Catoni and Giulini (2017)
    '''
    return np.where((np.abs(u) > math.sqrt(2)),\
                    (np.abs(u)*2*math.sqrt(2)/3-0.5),\
                    (u**2/2-u**4/24))

def psi_catnew(u):
    '''
    Influence function of Catoni and Giulini (2017).
    '''
    return np.where((np.abs(u) > math.sqrt(2)),\
                    (np.sign(u)*2*math.sqrt(2)/3),\
                    (u-u**3/6))

CONST_catnew = math.sqrt(81/32)


def psi_gud(u):
    '''
    Gudermannian function.
    '''
    return 2 * np.arctan(np.exp(u)) - np.pi/2


def est_gud(x, s, thres=1e-03, iters=50):
    '''
    M-estimate of location using Gudermannian function.
    '''
    
    new_theta = np.mean(x)
    old_theta = None
    diff = 1.0

    # Solve the psi-condition.
    for t in range(iters):
        old_theta = new_theta
        new_theta = old_theta + s * np.mean(psi_gud((x-old_theta)/s))
        diff = abs(old_theta-new_theta)
        if diff <= thres:
            break
            
    return new_theta


def normCDF(u):
    '''
    New, faster version of the Normal CDF computation,
    using scipy. This was found to be almost 40x faster
    than a naive vectorize implementation.
    '''
    return (1 + spec.erf(u/math.sqrt(2))) / 2


def r_catnew(m, sigma):
    '''
    The correction term used in computing the
    varphi function.
    '''
    # Some quantities to save.
    vm = (math.sqrt(2) - m) / sigma
    vp = (math.sqrt(2) + m) / sigma
    Fm = normCDF(-vm)
    Fp = normCDF(-vp)
    em = np.exp(-vm**2/2)
    ep = np.exp(-vp**2/2)

    # Broken up into five terms.
    t1 = 2*math.sqrt(2) * (Fm - Fp) / 3

    t2 = (-1) * (m-m**3/6) * (Fm + Fp)

    t3 = sigma * (1-m**2/2) * (ep - em) / math.sqrt(2*math.pi)

    t4 = m * sigma**2 * (Fp+Fm+(vp*ep + vm*em)/math.sqrt(2*math.pi)) / 2

    t5 = sigma**3 * ((2+vm**2)*em-(2+vp**2)*ep) / (6*math.sqrt(2*math.pi))

    # Just sum them up and return.
    return t1 + t2 + t3 + t4 + t5


def est_catnew(x, lam, beta):
    '''
    This is a matrix version of est_catnew_flat:
     - Assume x has shape (n,k)
     - Assumes that rows correspond to distict observations.
     - Assumes lam and beta have shape (k,) or (), can be just scalars.
    '''

    # Shape checks.
    n, k = x.shape
    if len(x.shape) < 2:
        raise ValueError("Shapes are not as expected.")
    
    # Main computations have no issues with under/overflow.
    comps = x * (1 - (lam*x)**2/(2*beta)) - (lam**2)*(x**3)/6
    
    # Make sure things are numerically stable for corrections.
    # note: beta is safe as-is.
    lam_safe = np.clip(a=lam, a_min=cf._lam_min, a_max=None)
    
    # Final computations based on safe values.
    corr = r_catnew(
        m=lam_safe*x,
        sigma=np.where((lam_safe*np.abs(x) < cf._sigma_min),
                       cf._sigma_min,
                       lam_safe*np.abs(x))/np.sqrt(beta)
    ) / lam_safe
    out_comps = np.mean(comps, axis=0)
    out_corr = np.mean(corr, axis=0)
    return out_comps + out_corr


def catoni_upbd(x):
    '''
    Key upper bound on Catoni-type influence functions.
    '''
    return np.log1p(x+x**2/2)


def catoni_lowbd(x):
    '''
    Key lower bound on Catoni-type influence functions.
    '''
    return -np.log1p(-x+x**2/2)


## FUNCTIONS FOR COMPUTING DISTRIBUTION PARAMETERS. ##

def vlnorm(meanlog, sdlog):
    '''
    Variance of the log-Normal distribution.
    '''
    return (math.exp(sdlog**2) - 1) * math.exp((2*meanlog + sdlog**2))


def mlnorm(meanlog, sdlog):
    '''
    Mean of log-Normal distribution.
    '''
    return math.exp((meanlog + sdlog**2/2))


def mnt_weibull(m, shape, scale):
    '''
    mth moment of the Weibull(shape,scale) distribution.
    '''
    return scale**m * spec.gamma(1+m/shape)


## CLERICAL FUNCTIONS. ##

def makedir_safe(dirname):
    '''
    A simple utility for making new directories
    after checking that they do not exist.
    '''
    if not os.path.exists(dirname):
        os.makedirs(dirname)






