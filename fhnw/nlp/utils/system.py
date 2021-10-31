def install(package, version=None):
    """Programmatically installes a package using pip

    Parameters
    ----------
    package : str
        The package to install
    version : str
        A specific version number
    """
        
    import pip
    
    try:
        __import__(package)
    except:
        import sys
        import subprocess
        
        install_package = package
        if version is not None:
            install_package + "==" + version
            
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', install_package])
        __import__(package) 
        
def gpu_empty_cache():
    """Cleans the GPU cache which seems to fill up after a while
    
    """
        
    import torch
    import tensorflow as tf

    if tf.config.list_physical_devices("GPU"):
        torch.cuda.empty_cache()
    
def get_gpu_device_number():
    """Provides the number of the GPU device
    
    Returns
    -------
    int
        The GPU device number of -1 if none is installed
    """
        
    import tensorflow as tf
    
    return 0 if tf.config.list_physical_devices("GPU") else -1

def get_compute_device():
    """Provides the device for the computation
    
    Returns
    -------
    str
        The GPU device with number (cuda:0) of cpu
    """
        
    import tensorflow as tf
    
    return "cuda:0" if tf.config.list_physical_devices("GPU") else "cpu"
