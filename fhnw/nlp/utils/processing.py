import psutil
import time
from multiprocess import Pool
from functools import partial

import pickle
import numpy as np
import pandas as pd

def parallelize_dataframe(df, func, **func_params):
    """Breaks a pandas dataframe in n_core parts and spawns n_cores processes to apply the provided function to all the parts

    Parameters
    ----------
    df : dataframe
        The dataframe with the data
    func : function
        The function to apply
    func_params : The function parameters
        The parameters for the function. This can contain 'n_cores' to specify the exact number of processes to spawn (otherwise 'psutil.cpu_count(logical=True)') is used) and 'field_read' to specify the data column (and of course all the parameters needed by the provided function)

    Returns
    -------
    dataframe
        A dataframe with the result of the function call (or the original dataframe concatenated with the processing result in case 'field_read' was not provided)
    """

    # https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
    # https://www.digitalocean.com/community/tutorials/how-to-use-args-and-kwargs-in-python-3
    # https://stackoverflow.com/a/28975239
    # http://python.omics.wiki/multiprocessing_map/multiprocessing_partial_function_multiple_arguments
    # https://stackoverflow.com/questions/45545110/make-pandas-dataframe-apply-use-all-cores
    # https://github.com/jmcarpenter2/swifter (an alternative)

    n_cores = func_params.pop("n_cores", -1)
    if n_cores <= 0:
        n_cores = psutil.cpu_count(logical=True)

    # do not grow the dataframe directly - see https://stackoverflow.com/a/56746204
    field_read = func_params.get("field_read")
    if field_read is not None:
        read_df = df[field_read].to_frame(field_read)
    else:
        read_df = df

    function_with_params = partial(func, **func_params)
    df_split = np.array_split(read_df, n_cores)

    pool = Pool(n_cores)
    write_df = pd.concat(pool.map(function_with_params, df_split))
    pool.close()
    pool.join()

    if field_read is not None:
        # df_ret = df.assign(field_write = write_df[field_write].values)
        return pd.concat([df, write_df], axis=1)
    else:
        return write_df


def is_iterable(obj):
    """Checks if an object is iterable

    Parameters
    ----------
    obj : object
        The object to check if it is iterable
        
    Returns
    -------
    bool
        True if the object is iterable, False otherwise
    """
    
    try:
        iter(obj)
        return True
    except TypeError:
        return False
