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
    
    if torch.cuda.is_available():
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
        The GPU device with number (cuda:0) or cpu
    """
    
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def system_info(): 
    """Provides some system information like OS etc.
    
    Returns
    -------
    str
        The system information
    """
      
    import os
    s = "OS name: "+ str(os.name)
    
    import platform
    s = s + os.linesep + "Platform name: " + str(platform.system())
    s = s + os.linesep + "Platform release: " + str(platform.release())
    s = s + os.linesep + "Python version: "+ str(platform.python_version())
    
    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        s = s + os.linesep + "CPU brand: "+ str(cpu_info["brand_raw"])
    except ImportError as e:
        pass
    except Exception as e:
    	print("Error in 'cpuinfo':", e)
    	pass
        
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
        s = s + os.linesep + "CPU cores: "+ str(cores)
        memory = psutil.virtual_memory()
        s = s + os.linesep +"RAM: " + str(round(memory.total/(1024.**3), 2)) + "GB total and " + str(round(memory.available/(1024.**3), 2)) + "GB available"
    except ImportError as e:
        pass
    except Exception as e:
    	print("Error in 'psutil':", e)
    	pass

    gpu_available = False
    try:
        import tensorflow as tf
        s = s + os.linesep + "Tensorflow version: " + str(tf.__version__)
        
        gpu_available = tf.config.list_physical_devices("GPU")
        s = s + os.linesep + "GPU is " + ("available" if gpu_available else "NOT AVAILABLE")
    except ImportError as e:
        pass
    except Exception as e:
    	print("Error in 'tensorflow':", e)
    	pass
        
    try:
        import igpu

        gpu_count = igpu.count_devices()
        for i in range(gpu_count):
           gpu = igpu.get_device(i)
           s = s + os.linesep + "GPU is a "+ str(gpu.name) +" with "+ str(round(gpu.memory.total)) + str(gpu.memory.unit)
    except ImportError as e:
        pass
    except Exception as e:
    	print("Error in 'igpu':", e)
    	pass
    
    return s
    
    
def set_log_level(level = 1):
    """Sets the log level of python/tensorflow

    Parameters
    ----------
    level : int
        The log level (0 -> NOTSET, 1 -> WARNING, 2 -> ERROR, 3 -> CRITICAL
    version : str
        A specific version number
    """
    
    import logging
    import os

    tf_level = logging.WARN
    if level == 0:
        tf_level =  logging.NOTSET
    elif level == 1:
        tf_level = logging.WARNING
    elif level == 2:
        tf_level = logging.ERROR
    else:
        tf_level = logging.CRITICAL
    
    # see https://stackoverflow.com/a/42121886
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(level)
    # see https://docs.python.org/3/library/logging.html#logging-levels
    logging.getLogger("tensorflow").setLevel(tf_level)
