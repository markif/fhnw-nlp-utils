
def join_tokens(tokens, stopwords = set()):
    """Joins tokens to a string

    Parameters
    ----------
    tokens : iterable
        The tokens to join to a string
    stopwords : set
        A set of stopword to ignore when joining the tokens (default is an empty set)
        
    Returns
    -------
    str
        The joined tokens
    """
    
    if not stopwords:
        return " ".join(tokens)
    else:
        return " ".join(token for token in tokens if token not in stopwords)
        

def clean_text(text, keep_punctuation=False, ensure_whitespace_after_punctuation=True):
    """Cleans text by removing html tags, non ascii chars, digits and optionally punctuation

    Parameters
    ----------
    text : str
        The text to clean
    keep_punctuation : bool
        Defines if punctuation should be kept
    ensure_whitespace_after_punctuation : bool
        Defines if a whitespace should be added after punctuation (.,;:!?) if one is missing
        
    Returns
    -------
    str
        The cleaned text
    """
    import re
    RE_TAGS = re.compile(r"<[^>]+>")
    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
    
    # remove any html tags (< /br> often found)
    text = re.sub(RE_TAGS, " ", text)
    
    if ensure_whitespace_after_punctuation:
        RE_SPACE_AFTER_PUNKT = re.compile(r"(?<=[.,;:!?])(?=[A-Za-z])")
        text = re.sub(RE_SPACE_AFTER_PUNKT, " ", text)
    
    if keep_punctuation:
        RE_ASCII_PUNCTUATION = re.compile(r"[^A-Za-zÀ-ž,.!? ]", re.IGNORECASE)
        RE_SINGLECHAR_PUNCTUATION = re.compile(r"\b[A-Za-zÀ-ž,.!?]\b", re.IGNORECASE)
    
        # keep only ASCII + European Chars and whitespace, no digits, keep punctuation
        text = re.sub(RE_ASCII_PUNCTUATION, " ", text)
        # convert all whitespaces (tabs etc.) to single wspace, keep punctuation
        text = re.sub(RE_SINGLECHAR_PUNCTUATION, " ", text)
    else:
        RE_ASCII = re.compile(r"[^A-Za-zÀ-ž ]", re.IGNORECASE)
        RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE)
        
        # keep only ASCII + European Chars and whitespace, no digits, no punctuation
        text = re.sub(RE_ASCII, " ", text)
        # convert all whitespaces (tabs etc.) to single wspace
        text = re.sub(RE_SINGLECHAR, " ", text)     
    
    text = re.sub(RE_WSPACE, " ", text)  
    return text


def create_ngram_counts(df, n = 2, field_read="token_lemma"):
    """Creates the n-gram counts of a column of text tokens

    Parameters
    ----------
    df : dataframe
        The dataframe
    n : int
        The n grams to create (e.g. 2 for bi-gram)
    field_read : str
        The column name to read from (default is token_lemma)
        
    Returns
    -------
    Counter
        The n-gram counts
    """
    from collections import Counter
        
    counter = Counter()

    # see https://stackoverflow.com/a/17071908
    #_ = df[field_read].apply(lambda x: counter.update(nltk.ngrams(x, n)))
    #_ = df[field_read].apply(lambda x: counter.update([" ".join(grams) for grams in nltk.ngrams(x, n)]))
    _ = df[field_read].apply(lambda x: counter.update([" ".join(x[i:i+n]) for i in range(len(x)-n+1)]))
    
    return counter
