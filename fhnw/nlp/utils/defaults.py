class NLPUtilsDefaults():
    
    def __init__(self):
        import multiprocess as mp
        
        self.lock = mp.Manager().Lock()
        self.stemmer = None
        self.tokenizer = None
        self.compound_word_splitter = None
    
__nlp_utils_defaults = NLPUtilsDefaults()


def default_stemmer():
    """Initialization of default stemmer
    
    Returns
    -------
    callable
        The default stemmer (nltk SnowballStemmer)
    """
    
    if __nlp_utils_defaults.stemmer is None:
        with __nlp_utils_defaults.lock:
            if __nlp_utils_defaults.stemmer is None:
                import nltk
                from nltk.stem.snowball import SnowballStemmer
    
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt')

                __nlp_utils_defaults.stemmer = SnowballStemmer("german")
                
    return __nlp_utils_defaults.stemmer.stem


def default_tokenizer():
    """Initialization of default tokenizer
    
    Returns
    -------
    callable
        The default tokenizer (nltk word_tokenize)
    """
        
    if __nlp_utils_defaults.tokenizer is None:
        with __nlp_utils_defaults.lock:
            if __nlp_utils_defaults.tokenizer is None:
                import nltk
                from nltk.tokenize import word_tokenize

                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt')

                __nlp_utils_defaults.tokenizer = word_tokenize
                
    return __nlp_utils_defaults.tokenizer


def default_compound_splitter():
    """Initialization of default compound word splitter
    
    Returns
    -------
    callable
        The default compound word splitter (compound_split)
    """
    
    if __nlp_utils_defaults.compound_word_splitter is None:
        with __nlp_utils_defaults.lock:
            if __nlp_utils_defaults.compound_word_splitter is None:
                from compound_split import doc_split

                __nlp_utils_defaults.compound_word_splitter = doc_split.maximal_split
                
    return __nlp_utils_defaults.compound_word_splitter
