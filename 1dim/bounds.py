
# bounds.py

import numpy as np
from scipy import special

import config as cf


def mom(n, var, delta):
    '''
    See Devroye et al. (Ann Stats, 2016), Thm 4.1.
    '''
    return 2*np.sqrt(2*np.exp(1)) * np.sqrt(var*(1+np.log(1/delta))/n)


def mest(n, var, delta):
    '''
    See, for example, Holland and Ikeda (ICML 2019).
    '''
    return 2*np.sqrt(2*var*np.log(1/delta)/n)


def mult_bernoulli(n, mnt, delta):
    
    return np.sqrt(2*mnt*np.log(1/delta)/n)


def mult_bernoulli_centered(n, mnt, var, delta):

    n_center = n//cf._n_center_factor
    n_main = n-n_center
    bound_center = np.sqrt(2*mnt*np.log(1/delta)/n_center)
    mnt_main = var + bound_center**2
    bound_main = np.sqrt(2*mnt_main*np.log(2/delta)/n_main)
    # note: have 2/delta here since the union of the
    # two good events holds with prob no less than 1-2*delta.
    # Thus, the bound returned here is one that holds with
    # probability no less than 1-delta.
    return bound_main


def mult_gaussian(n, mnt, delta):
    
    return np.sqrt(2*mnt*np.log(1/delta)/n) + np.sqrt(mnt/n)


def add_gaussian(n, mnt, delta):
    
    return np.sqrt(2*mnt*np.log(1/delta)/n) + 1/np.sqrt(n)


def mult_weibull(n, mnt, delta):
    
    k = cf._mult_weibull_k
    gamval = special.gamma(1+1/k)
    c = 1/gamval**k + k*np.log(gamval) - 1
    bb = 2*special.gamma(1+2/k)*mnt*(c+np.log(1/delta)) / (gamval**2 * n)
    return np.sqrt(bb)


def add_weibull(n, mnt, delta):
    
    sigma = cf._add_weibull_sigma
    bb = 2 * (mnt+sigma**2*(1-np.pi/4)) * np.log(1/delta) / n # assumes k=2.0, as in paper.
    return np.sqrt(bb)


def mult_student(n, mnt, delta):
    
    df = cf._add_student_df
    gamquot = special.gamma((df+1)/2) / special.gamma(df/2)
    c = ((df+1)/2) * ( np.log(1+1/df**2) + 4*gamquot/(np.sqrt(df*np.pi)*(df-1)) ) # assumes a=1.
    bb = mnt * ((df-1)/(df-2)) * (c+np.log(1/delta)) / n
    return 2*np.sqrt(bb)


def add_student(n, mnt, delta):
    
    df = cf._add_student_df
    gamquot = special.gamma((df+1)/2) / special.gamma(df/2)
    c = ((df+1)/2) * ( np.log(1+1/df**2) + 4*gamquot/(np.sqrt(df*np.pi)*(df-1)) ) # assumes a=1.
    bb = 2 * (mnt+df/(df-2)) * (c+np.log(1/delta)) / n
    return np.sqrt(bb)


