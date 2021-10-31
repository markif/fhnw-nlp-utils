
def get_filename(path):
    """Extracts the filename from a string

    Parameters
    ----------
    path : str
        The path  

    Returns
    -------
    str
        The filename
    """
    
    import ntpath
    
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_path(path):
    """Extracts the path (without file) from a string

    Parameters
    ----------
    path : str
        The path  

    Returns
    -------
    str
        The path without file
    """
    
    import ntpath
    
    head, tail = ntpath.split(path)
    return head

def save_dataframe(df, path):
    """Saves a dataframe to disk

    Parameters
    ----------
    df : dataframe
        The dataframe to save
    path : str
        The path to store the dataframe (if the path does not exist it will be created). The ending of the file defines the storage type. csv -> store the data as csv in a zip compressed archive, zip -> store the data as csv in a zip compressed archive, parq -> brötli compressed parquet file format.
    """
    
    import os
    from pathlib import Path
    import pandas
    
    filename = get_filename(path)
    archive_name = filename
        
    if filename.endswith(".csv"):
        filename = os.path.splitext(filename)[0] + ".zip"
    elif filename.endswith(".zip"):
        archive_name = os.path.splitext(filename)[0] + ".csv"
    elif filename.endswith(".parq"):
        filename = filename
    else:
        filename = filename + ".zip"
        archive_name = filename + ".csv"

    parent_path = get_path(path)
    # create if not exists
    Path(parent_path).mkdir(parents=True, exist_ok=True)
    path = os.path.join(parent_path, filename)
   
    if filename.endswith(".parq"):
        df.to_parquet(path, compression='brotli')
    else:
        compression_opts = dict(method='zip',archive_name=archive_name)
        df.to_csv(path, index=False, compression=compression_opts)
    
def load_dataframe(path):
    """Loads a dataframe from disk

    Parameters
    ----------
    path : str
        The path of the file to load the dataframe from. Supported formats are: csv in a zip compressed archive, parq as parquet file format.
        
    Returns
    -------
    dataframe
        The loaded dataframe
    """
    
    import os
    import errno
    import pandas
    
    if not os.path.exists(path):
        filename = get_filename(path)
        try_path = path
        
        if filename.endswith(".csv"):
            filename = os.path.splitext(filename)[0] + ".zip"
            try_path = os.path.join(get_path(path), filename)

        if not os.path.exists(try_path):
            try_path = path  + ".zip"

        if not os.path.exists(try_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        else:
            path = try_path
            
    if path.endswith(".parq"):
        return pandas.read_parquet(path)
    else:
        return pandas.read_csv(path)

def save_dataframe_pickle(df, path):
    """Saves a pickled dataframe to disk

    Parameters
    ----------
    df : dataframe
        The dataframe to save
    path : str
        The path to store the dataframe (if the path does not exist it will be created).
    """
    
    from pathlib import Path
    import pandas
    
    parent_path = get_path(path)
    # create if not exists
    Path(parent_path).mkdir(parents=True, exist_ok=True)

    df.to_pickle(path)
    
def load_dataframe_pickle(path):
    """Loads a pickled dataframe from disk

    Parameters
    ----------
    path : str
        The path of the file to load the dataframe from.
        
    Returns
    -------
    dataframe
        The loaded dataframe
    """

    return pandas.read_pickle(path)


def download(url, path, re_download=False):
    """Download data from an url and stores it in a file

    Parameters
    ----------
    url : str
        The url to the data
    path: str
        The path to store the data
    re_download: bool
        Forces a re-download even if the file already exists
    """
    
    import os
    import wget
    import gdown
    from pathlib import Path
    
    parent_path = get_path(path)
    # create if not exists
    Path(parent_path).mkdir(parents=True, exist_ok=True)
    
    if re_download or not os.path.exists(path):
        if "drive.google.com" in url:
            gdown.download(url, path, quiet=True)
        else:
            try:
                wget.download(url, out=path)
            except:
                # fallback to system
                try:
                    p = os.system("wget " + url +" -O "+path)
                    p.wait()
                except:
                    raise LookupError("Download failed. You probably need to install 'wget' on your system/docker env.")

