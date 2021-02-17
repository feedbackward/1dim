
# data.py

import numpy as np

import config
import helpers as hlp


def data_normal(n, mean, sd):
    return np.random.normal(loc=mean, scale=sd, size=n)


def data_lognormal(n, meanlog, sdlog):
    return np.random.lognormal(mean=meanlog, sigma=sdlog, size=n)


def parse_data(distro, level, mean):

    true_paras = {}
    
    if distro == "normal":
        sd = config._sd[level]
        true_paras["mean"] = mean
        true_paras["var"] = sd**2
        true_paras["mnt"] = true_paras["var"] + true_paras["mean"]**2
        
        fn = lambda m : data_normal(n=m, mean=mean, sd=sd)
        
        return (fn, true_paras)
    
    elif distro == "lognormal":
        meanlog = config._meanlog
        sdlog = config._sdlog[level]
        # subtract initial mean and shift.
        mean_shift = mean - hlp.mlnorm(meanlog=meanlog, sdlog=sdlog)
        true_paras["mean"] = mean
        true_paras["var"] = hlp.vlnorm(meanlog=meanlog, sdlog=sdlog)
        true_paras["mnt"] = true_paras["var"] + true_paras["mean"]**2
        
        fn = lambda m : data_lognormal(n=m,
                                       meanlog=meanlog,
                                       sdlog=sdlog)+mean_shift # shifting.
        
        return (fn, true_paras)
    
    
    
def parse_data_ratio(distro, level, ratio):

    true_paras = {}
    
    if distro == "normal":
        sd = config._sd[level]
        true_paras["var"] = sd**2
        true_paras["mean"] = ratio*np.sqrt(true_paras["var"])
        true_paras["mnt"] = true_paras["var"] + true_paras["mean"]**2
        
        fn = lambda m : data_normal(n=m, mean=true_paras["mean"], sd=sd)
        
        return (fn, true_paras)
    
    elif distro == "lognormal":
        meanlog = config._meanlog
        sdlog = config._sdlog[level]
        # subtract initial mean and shift.
        true_paras["var"] = hlp.vlnorm(meanlog=meanlog, sdlog=sdlog)
        true_paras["mean"] = ratio*np.sqrt(true_paras["var"])
        mean_shift = true_paras["mean"] - hlp.mlnorm(meanlog=meanlog, sdlog=sdlog)
        true_paras["mnt"] = true_paras["var"] + true_paras["mean"]**2
        
        fn = lambda m : data_lognormal(n=m,
                                       meanlog=meanlog,
                                       sdlog=sdlog)+mean_shift # shifting.
        
        return (fn, true_paras)
        
        
def get_mnt(distro, level, mean):
    '''
    Just a simple wrapper.
    '''
    return parse_data(distro=distro, level=level, mean=mean)[1]["mnt"]
    
