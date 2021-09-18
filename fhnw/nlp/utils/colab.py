
def runs_on_colab():
    """Determines if the working environment is google colab
    """
    
    return 'google.colab' in str(get_ipython())
