
# main.py

import multiprocessing as mltp
import os
import numpy as np

import config as cf
import data
import methods
import helpers as hlp


# Prepare a results folder, if doesn't already exist.
hlp.makedir_safe("results")


def worker(mth_name):

    for (distro, level, n) in cf._condition_grid:

            perf = np.zeros((cf._ratios_len, cf._num_trials),
                            dtype=np.float32) # to fill with results.

            for k in range(cf._ratios_len):

                ratio = cf._ratios[k]
                
                data_fn, true_paras = data.parse_data_ratio(distro=distro,
                                                            level=level,
                                                            ratio=ratio)
                mth_paras = {"n": n,
                             "mnt": true_paras["mnt"],
                             "var": true_paras["var"],
                             "delta": cf._delta}
                mth = methods.parse_mth(mth_name=mth_name,
                                        paras=mth_paras)
                
                print(
                    "mth {}, distro {}, level {}, mean {}, n {}".format(mth_name,
                                                                        distro,
                                                                        level,
                                                                        true_paras["mean"], n)
                )
                
                # Run a loop over all trials.
                
                for t in range(cf._num_trials):

                    x = data_fn(m=n)
                    xhat = mth(u=x)
                    perf[k,t] = np.abs(true_paras["mean"]-xhat)


            # After all the loops have finished and the perf matrix is
            # completely full, then all that remains is to write to file.
            filename = methods.perf_filename(mth_name=mth_name,
                                             distro=distro,
                                             level=level, n=n)
            towrite = os.path.join("results", filename)
            np.savetxt(fname=towrite, X=perf, fmt="%.7e", delimiter=",")
        
# End of worker definition.


if __name__ == "__main__":
    
    cpu_count = mltp.cpu_count()
    print("Our machine has", cpu_count, "CPUs.")
    print("Of these,", len(os.sched_getaffinity(0)), "are available.")
    
    # Put all processors to work (at an upper limit).
    mypool = mltp.Pool(cpu_count)
    
    # Pass the "worker" the name of the algorithm to run.
    mypool.map(func=worker, iterable=cf._mth_names)
    
    # Memory management.
    mypool.close() # important for stopping memory leaks.
    mypool.join() # wait for all workers to exit.

# End of multiproc procedure.
