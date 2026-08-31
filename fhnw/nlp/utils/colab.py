
def runs_on_colab():
    """Determines if the working environment is google colab
    """
    
    return 'google.colab' in str(get_ipython())


def runs_on_azure():
    """Determines if the working environment is azure ml
    """

    return 'azureuser' in str(get_ipython().home_dir)  # True

