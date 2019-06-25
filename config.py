
# config.py

import numpy as np
import itertools


# Figure-related parameter.
_fontsize = "xx-large"

# Method-related parameters.
_mth_names = ["mean", "med", "mom", "mest",
              "mult_g", "mult_b", "mult_bc", "mult_w", "mult_s",
              "add_g", "add_w", "add_s"]

_mth_colors = {"mean":"xkcd:bordeaux", "med":"xkcd:dull yellow",
               "mom":"xkcd:hot pink", "mest":"xkcd:violet",
               "mult_g":"xkcd:ultramarine", "add_g":"xkcd:periwinkle",
               "mult_b":"black", "mult_bc":"xkcd:warm grey",
               "mult_w":"xkcd:grass green", "add_w":"xkcd:drab green",
               "mult_s":"xkcd:orange", "add_s":"xkcd:pale orange"}

_delta = 0.01 # confidence level parameter.
_n_center_factor = 2 # factor to do sample-splitting when centering.
_mult_weibull_k = 2.0 # anything is fine.
_add_weibull_k = 2.0 # following paper.
_add_weibull_sigma = 1.0 # anything is fine.
_mult_student_df = 5.1 # need to have larger than 5.0 to compute.
_add_student_df = 5.1 # need to have larger than 5.0 to compute.
_lam_min = 1e-3 # for numerical stability.
_sigma_min = 1e-4 # for numerical stability.

# Experimental conditions.
_num_trials = 10000
_nvals = [10,20,30,40,50,60,70,80,90,100]
_nvals_len = len(_nvals)
_distros = ["normal", "lognormal"]
_distros_len = len(_distros)
_levels = ["low", "mid", "high"]
_levels_len = len(_levels)
_ratios = np.arange(-2.0, 2.0+0.05, 0.05)
_ratios_len = _ratios.size
_condition_grid = itertools.product(_distros, _levels, _nvals)

# Normal parameters.
_sd = {"low": 0.5, "mid": 5.0, "high": 50.0}

# log-Normal parameters
_meanlog = 0
_sdlog = {"low": 1.1, "mid": 1.35, "high": 1.75}

