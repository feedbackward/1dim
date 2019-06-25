
# methods.py

import numpy as np
from scipy import stats
from scipy import special

import config as cf
import helpers as hlp


## Implementations of the F_0, F_3, and F_W functions. ##

def F_0(a, b, M_0):
    '''
    This function is executed assuming b != 0 (see F_W defn).
    '''
    
    sq = np.sqrt(2)
    c = 2*sq / 3
    return c + M_0(sq, a=a, b=b) * (a-c-a**3/6) - M_0(-sq, a=a, b=b) * (a+c-a**3/6)

def F_3(a, b, M_1, M_2, M_3):
    '''
    This function is executed assuming b != 0 (see F_W defn).
    '''
    
    sq = np.sqrt(2)
    D_1 = M_1(sq, a=a, b=b)-M_1(-sq, a=a, b=b)
    D_2 = M_2(sq, a=a, b=b)-M_2(-sq, a=a, b=b)
    D_3 = M_3(sq, a=a, b=b)-M_3(-sq, a=a, b=b)
    return (a**2/2-1)*b*D_1 + (a*b**2/2)*D_2 + (b**3/6)*D_3

def F_W(a, b, M_0, M_1, M_2, M_3):
    '''
    F_W function, the top level of computation.
    When b=0, value is immediately phi(a).
    '''
    
    return np.where(b==0.,
                    hlp.psi_catnew(u=a),
                    F_0(a=a, b=b, M_0=M_0)-F_3(a=a, b=b, M_1=M_1, M_2=M_2, M_3=M_3))


## Implementations of the M^k functions. ##

# Gaussian

def M_0_gaussian(x, a, b, paras=None):
    
    z = (x-a)/b
    return stats.norm.cdf(z)

def M_1_gaussian(x, a, b, paras=None):
    
    z = (x-a)/b
    return -np.exp(-z**2/2) / np.sqrt(2*np.pi)

def M_2_gaussian(x, a, b, paras=None):
    
    z = (x-a)/b
    return M_0_gaussian(x=x, a=a, b=b) + z * M_1_gaussian(x=x, a=a, b=b)

def M_3_gaussian(x, a, b, paras=None):
    
    z = (x-a)/b
    return (z**2+2) * M_1_gaussian(x=x, a=a, b=b)


# Weibull

def M_0_weibull_core(z, paras):
    '''
    Core computations for Weibull case.
    Note that we take abs(z) because when
    z is negative, the value to be returned
    is trivially zero (see final condition).
    Similar deal for M_1, M_2, M_3.
    '''
    k = paras["shape"]
    sigma = paras["scale"]
    topass = (np.abs(z)/sigma)**k
    return np.where(z<=0., 0., 1-np.exp(-topass))
    
def M_0_weibull(x, a, b, paras):
    
    z = (x-a)/b
    toreturn = M_0_weibull_core(z=z, paras=paras)
    return np.where(b>=0., toreturn, 1-toreturn)

def M_1_weibull_core(z, paras):
    
    k = paras["shape"]
    sigma = paras["scale"]
    topass = (np.abs(z)/sigma)**k
    gamval = special.gammainc(1/k, topass) * special.gamma(1/k)
    outpos = (sigma/k) * gamval - z * np.exp(-topass)
    return np.where(z <= 0., 0., outpos)

def M_1_weibull(x, a, b, paras):
    
    z = (x-a)/b
    toreturn = M_1_weibull_core(z=z, paras=paras)
    mntval = hlp.mnt_weibull(m=1, shape=paras["shape"], scale=paras["scale"])
    return np.where(b>=0., toreturn, mntval-toreturn)

def M_2_weibull_core(z, paras):
    
    k = paras["shape"]
    sigma = paras["scale"]
    topass = (np.abs(z)/sigma)**k
    gamval = special.gammainc(2/k, topass) * special.gamma(2/k)
    outpos = (2*sigma**2/k) * gamval - z**2 * np.exp(-topass)
    return np.where(z <= 0., 0., outpos)

def M_2_weibull(x, a, b, paras):
    
    z = (x-a)/b
    toreturn = M_2_weibull_core(z=z, paras=paras)
    mntval = hlp.mnt_weibull(m=2, shape=paras["shape"], scale=paras["scale"])
    return np.where(b>=0., toreturn, mntval-toreturn)

def M_3_weibull_core(z, paras):
    
    k = paras["shape"]
    sigma = paras["scale"]
    topass = (np.abs(z)/sigma)**k
    gamval = special.gammainc(3/k, topass) * special.gamma(3/k)
    outpos = (3*sigma**3/k) * gamval - z**3 * np.exp(-topass)
    return np.where(z <= 0., 0., outpos)

def M_3_weibull(x, a, b, paras):
    
    z = (x-a)/b
    toreturn = M_3_weibull_core(z=z, paras=paras)
    mntval = hlp.mnt_weibull(m=3, shape=paras["shape"], scale=paras["scale"])
    return np.where(b>=0., toreturn, mntval-toreturn)
    

# Student-t

def M_0_student(x, a, b, paras=None):
    
    df = paras["df"]
    z = (x-a)/b
    return stats.t.cdf(z, df=df)

def A_k(df):
    
    return special.gamma((df+1)/2) / (np.sqrt(df*np.pi)*special.gamma(df/2))

def M_1_student(x, a, b, paras=None):
    
    df = paras["df"]
    z = (x-a)/b
    A = A_k(df=df)
    return (df/(1-df)) * A / (1 + z**2/df )**((df-1)/2)

def M_2_student(x, a, b, paras=None):
    
    df = paras["df"]
    z = (x-a)/b
    A = A_k(df=df)
    A_m2 = A_k(df=df-2)
    c = A * df**(3/2) / (np.sqrt(df-2)*(df-1))
    M_0 = M_0_student(x=z*np.sqrt((df-2)/df), a=0, b=1, paras={"df":df-2})
    return (c/A_m2) * M_0 - c * z * np.sqrt((df-2)/df) / (1+z**2/df)**((df-1)/2)

def M_3_student(x, a, b, paras=None):
    
    df = paras["df"]
    z = (x-a)/b
    A = A_k(df=df)
    A_m2 = A_k(df=df-2)
    M_1 = M_1_student(x=z*np.sqrt((df-2)/df), a=0, b=1, paras={"df":df-2})
    return (df**2 * A / ((df-2)*(df-1))) * ( 2*M_1/A_m2 - ((df-2)/df) * z**2 / (1+z**2/df)**((df-1)/2) )


## Main routines for all the new estimators. ##

def xhat_mult_bernoulli(x, s):
    
    return s * np.mean(hlp.psi_catnew(x/s))


def xhat_mult_bernoulli_centered(x, s_center, s_main):

    n_center = x.size//cf._n_center_factor
    n_main = x.size-n_center

    xtilde = xhat_mult_bernoulli(x=x[0:n_center], s=s_center)
    xhat = xhat_mult_bernoulli(x=(x[n_center:]-xtilde), s=s_main)
    xhat += xtilde

    return xhat


def xhat_mult_gaussian(x, s, mnt):
    
    n = x.size
    beta = np.sqrt(n*mnt/s**2)
    
    a = x/s
    b = np.abs(x)/(np.sqrt(beta)*s)
    return s * np.mean(F_W(a=a, b=b,
                           M_0=M_0_gaussian, M_1=M_1_gaussian,
                           M_2=M_2_gaussian, M_3=M_3_gaussian))


def xhat_mult_gaussian_sanity(x, s, mnt):
    '''
    Sanity check derived using the
    formulas from Catoni and Giulini (2017).
    '''
    n = x.size
    beta = np.sqrt(n*mnt/s**2)
    return hlp.est_catnew(x=x.reshape((n,1)), lam=(1/s), beta=beta)


def xhat_add_gaussian(x, s):
    
    n = x.size
    beta = np.sqrt(n/s**2)
    a = x/s
    b = 1/(np.sqrt(beta)*s)
    return s * np.mean(F_W(a=a, b=b,
                           M_0=M_0_gaussian, M_1=M_1_gaussian,
                           M_2=M_2_gaussian, M_3=M_3_gaussian))


def xhat_mult_weibull(x, s):
    
    a = 0.
    b = x/s
    k = cf._mult_weibull_k
    sigma = 1 / special.gamma(1+1/k) # to control mean=1.0.
    paras = {"shape": k, "scale": sigma}
    M_0 = lambda x,a,b : M_0_weibull(x=x, a=a, b=b, paras=paras)
    M_1 = lambda x,a,b : M_1_weibull(x=x, a=a, b=b, paras=paras)
    M_2 = lambda x,a,b : M_2_weibull(x=x, a=a, b=b, paras=paras)
    M_3 = lambda x,a,b : M_3_weibull(x=x, a=a, b=b, paras=paras)
    return s * np.mean(F_W(a=a, b=b,
                           M_0=M_0, M_1=M_1,
                           M_2=M_2, M_3=M_3))

def xhat_add_weibull(x, s):
    
    k = cf._add_weibull_k
    sigma = cf._add_weibull_sigma
    a = (x-sigma*np.sqrt(np.pi)/2)/s
    b = 1/s
    paras = {"shape": k, "scale": sigma}
    M_0 = lambda x,a,b : M_0_weibull(x=x, a=a, b=b, paras=paras)
    M_1 = lambda x,a,b : M_1_weibull(x=x, a=a, b=b, paras=paras)
    M_2 = lambda x,a,b : M_2_weibull(x=x, a=a, b=b, paras=paras)
    M_3 = lambda x,a,b : M_3_weibull(x=x, a=a, b=b, paras=paras)
    return s * np.mean(F_W(a=a, b=b,
                           M_0=M_0, M_1=M_1,
                           M_2=M_2, M_3=M_3))


def xhat_mult_student(x, s):
    
    df = cf._mult_student_df
    a = x/s
    b = np.abs(x)/s
    paras = {"df": df}
    M_0 = lambda x,a,b : M_0_student(x=x, a=a, b=b, paras=paras)
    M_1 = lambda x,a,b : M_1_student(x=x, a=a, b=b, paras=paras)
    M_2 = lambda x,a,b : M_2_student(x=x, a=a, b=b, paras=paras)
    M_3 = lambda x,a,b : M_3_student(x=x, a=a, b=b, paras=paras)
    return s * np.mean(F_W(a=a, b=b,
                           M_0=M_0, M_1=M_1,
                           M_2=M_2, M_3=M_3))

def xhat_add_student(x, s):
    
    df = cf._add_student_df
    a = x/s
    b = 1/s
    paras = {"df": df}
    M_0 = lambda x,a,b : M_0_student(x=x, a=a, b=b, paras=paras)
    M_1 = lambda x,a,b : M_1_student(x=x, a=a, b=b, paras=paras)
    M_2 = lambda x,a,b : M_2_student(x=x, a=a, b=b, paras=paras)
    M_3 = lambda x,a,b : M_3_student(x=x, a=a, b=b, paras=paras)
    return s * np.mean(F_W(a=a, b=b,
                           M_0=M_0, M_1=M_1,
                           M_2=M_2, M_3=M_3))
    

## Optimal scaling parameter settings for each method. ##

def s_mest(n, var, delta):
    
    return np.sqrt(2*n*var/np.log(1/delta))


def s_mult_bernoulli(n, mnt, delta):
    
    return np.sqrt(n*mnt/(2*np.log(1/delta)))


def s_mult_gaussian(n, mnt, delta):
    
    return np.sqrt(n*mnt/(2*np.log(1/delta)))


def s_add_gaussian(n, mnt, delta):
    
    return np.sqrt(n*mnt/(2*np.log(1/delta)))


def s_mult_weibull(n, mnt, delta):
    
    k = cf._mult_weibull_k
    gamval = special.gamma(1+1/k)
    c = 1/gamval**k + k*np.log(gamval) - 1
    ss = n * special.gamma(1+2/k) * mnt / (2*gamval**2*(c+np.log(1/delta)))
    return np.sqrt(ss)


def s_add_weibull(n, mnt, delta):
    
    sigma = cf._add_weibull_sigma
    ss = n * (mnt+sigma**2*(1-np.pi/4)) / (2*np.log(1/delta))
    return np.sqrt(ss)


def s_mult_student(n, mnt, delta):
    
    df = cf._add_student_df
    gamquot = special.gamma((df+1)/2) / special.gamma(df/2)
    c = ((df+1)/2) * ( np.log(1+1/df**2) + 4*gamquot/(np.sqrt(df*np.pi)*(df-1)) ) # assumes a=1.
    ss = n * mnt * (df-1) / ((df-2)*(c+np.log(1/delta)))
    return np.sqrt(ss)


def s_add_student(n, mnt, delta):
    
    df = cf._mult_student_df
    gamquot = special.gamma((df+1)/2) / special.gamma(df/2)
    c = ((df+1)/2) * ( np.log(1+1/df**2) + 4*gamquot/(np.sqrt(df*np.pi)*(df-1)) ) # assumes a=1.
    ss = n * (mnt+df/(df-2)) / (2*(c+np.log(1/delta)))
    return np.sqrt(ss)


def k_mom(n, delta):
    '''
    Number of partitions based on
    confidence level, following 4.1 of
    Devroye et al. (Ann Stats, 2016).
    '''
    if delta < np.exp(1-n/2):
        # If below level for which we have
        # guarantees, then just use the mean.
        return n
    else:
        return int(np.ceil(np.log(1/delta)))


## Implementation of the benchmark references. ##

def xhat_mean(x):
    
    return np.mean(x)


def xhat_med(x):
    
    return np.median(x)


def xhat_mom(x, k):
    
    n = x.size
    m = n // k
    
    if m < 2:
        return np.mean(x)
    
    idx_shuf = np.random.choice(n, size=k*m, replace=False)
    return np.median(np.mean(x[idx_shuf].reshape((k,m)), axis=1))


def xhat_mest(x, s):
    
    return hlp.est_gud(x=x, s=s)


## Clerical functions. ##

def parse_mth(mth_name, paras):

    if mth_name == "mean":
        return lambda u : xhat_mean(x=u)
    
    elif mth_name == "med":
        return lambda u : xhat_med(x=u)

    elif mth_name == "mom":
        k = k_mom(n=paras["n"], delta=paras["delta"])
        return lambda u : xhat_mom(x=u, k=k)
    
    elif mth_name == "mest":
        s = s_mest(n=paras["n"], var=paras["var"], delta=paras["delta"])
        return lambda u : xhat_mest(x=u, s=s)
        
    elif mth_name == "mult_b":
        s = s_mult_bernoulli(n=paras["n"], mnt=paras["mnt"], delta=paras["delta"])
        return lambda u : xhat_mult_bernoulli(x=u, s=s)

    elif mth_name == "mult_bc":
        n = paras["n"]
        mnt = paras["mnt"]
        delta = paras["delta"]
        n_center = n//cf._n_center_factor
        n_main = n-n_center
        s_center = s_mult_bernoulli(n=n_center, mnt=mnt,
                                    delta=delta)
        bound_center = np.sqrt(2*mnt*np.log(1/delta)/n_center)
        mnt_main = paras["var"] + bound_center**2
        s_main = s_mult_bernoulli(n=n_main, mnt=mnt_main,
                                  delta=paras["delta"])
        
        return lambda u : xhat_mult_bernoulli_centered(x=u,
                                                      s_center=s_center,
                                                      s_main=s_main)
    
    elif mth_name == "mult_g":
        s = s_mult_gaussian(n=paras["n"], mnt=paras["mnt"], delta=paras["delta"])
        return lambda u : xhat_mult_gaussian(x=u, s=s, mnt=paras["mnt"])
    
    elif mth_name == "mult_w":
        s = s_mult_weibull(n=paras["n"], mnt=paras["mnt"], delta=paras["delta"])
        return lambda u : xhat_mult_weibull(x=u, s=s)
    
    elif mth_name == "mult_s":
        s = s_mult_student(n=paras["n"], mnt=paras["mnt"], delta=paras["delta"])
        return lambda u : xhat_mult_student(x=u, s=s)
    
    elif mth_name == "add_g":
        s = s_add_gaussian(n=paras["n"], mnt=paras["mnt"], delta=paras["delta"])
        return lambda u : xhat_add_gaussian(x=u, s=s)
    
    elif mth_name == "add_w":
        s = s_add_weibull(n=paras["n"], mnt=paras["mnt"], delta=paras["delta"])
        return lambda u : xhat_add_weibull(x=u, s=s)
    
    elif mth_name == "add_s":
        s = s_add_student(n=paras["n"], mnt=paras["mnt"], delta=paras["delta"])
        return lambda u : xhat_add_student(x=u, s=s)
    
    else:
        return None


def perf_filename(mth_name, distro, level, n):
    
    return "{}_{}_{}_n{}.perf".format(mth_name, distro, level, n)


