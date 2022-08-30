def preprocess(text, stopwords, do_compound_word_split = False):
    """Preprocesses the text (tokenization, stop word removal, lemmantization/stemming)

    Parameters
    ----------
    text : str
        The text to preprocess
    stopwords : set
        A set of stopword to remove from the tokens
    do_compound_word_split : bool
        Defines if compound word splitting should be applied
        
    Returns
    -------
    list
        The preprocessed and tokenized text
    """
    
    compound_splitter = None
    if do_compound_word_split:
        compound_splitter = _compound_split
       
    return _preprocess(text, stopwords, word_splitter=compound_splitter, stemmer=_stem)


def _preprocess(text, stopwords, word_splitter=None, stemmer=None):
    """Preprocesses the text (tokenization, stop word removal, lemmantization/stemming)

    Parameters
    ----------
    text : str
        The text to preprocess
    stopwords : set
        A set of stopword to remove from the tokens
    word_splitter: callable
        The word splitter to use (callable e.g. compound_split to split compound words) or None to disable word splitting
    stemmer: callable
        The stemmer to use (callable e.g. SnowballStemmer) or None to disable stemming
        
    Returns
    -------
    list
        The preprocessed and tokenized text
    """

    from fhnw.nlp.utils.text import clean_text
    from fhnw.nlp.utils.normalize import normalize
    
    text = clean_text(text)
    word_tokens = normalize(text, stopwords=stopwords, word_splitter=word_splitter, stemmer=stemmer)

    return word_tokens


def _compound_split(word):
    """Splits a compound word into its subwords

    Parameters
    ----------
    word : str
        The word to split
        
    Returns
    -------
    list
        The subwords
    """    
    
    from fhnw.nlp.utils.defaults import default_compound_splitter

    return default_compound_splitter()(word)


def _stem(word):
    """Stems a word into its root form

    Parameters
    ----------
    word : str
        The word to stem
        
    Returns
    -------
    str
        The stemmed word
    """    
    
    from fhnw.nlp.utils.defaults import default_stemmer
    
    return default_stemmer()(word)
