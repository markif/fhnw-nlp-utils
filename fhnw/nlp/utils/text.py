import re
import nltk
import pandas as pd
from collections import Counter

from fhnw.nlp.utils.processing import is_iterable

empty_stopwords = set()

def join_tokens(tokens, stopwords = empty_stopwords):
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

def join_tokens_df(df, stopwords = empty_stopwords, field_read = "token_clean", field_write = "text_clean"):
    """Joins a column of tokens of a dataframe to strings (primarily meant for parallel processing)

    Parameters
    ----------
    df : dataframe
        The dataframe
    stopwords : set
        A set of stopword to ignore when joining the tokens (default is an empty set)
    field_read : str
        The column name to read the tokens from (default is token_clean)
    field_write : str
        The column name to write the joined tokens to (default is field_write)
        
    Returns
    -------
    dataframe
        A dataframe with the joined tokens
    """

    # do not grow the dataframe directly - see https://stackoverflow.com/a/56746204
    series = df[field_read].map(
        lambda x: join_tokens(x, stopwords) if is_iterable(x) else ""
    )
    
    return series.to_frame(field_write)

RE_TAGS = re.compile(r"<[^>]+>")
RE_ASCII = re.compile(r"[^A-Za-zÀ-ž ]", re.IGNORECASE)
RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE)
RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)

RE_ASCII_PUNCTUATION = re.compile(r"[^A-Za-zÀ-ž,.!? ]", re.IGNORECASE)
RE_SINGLECHAR_PUNCTUATION = re.compile(r"\b[A-Za-zÀ-ž,.!?]\b", re.IGNORECASE)

def clean_text(text, keep_punctuation=False):
    """Cleans text by removing html tags, non ascii chars, digits and optionally punctuation

    Parameters
    ----------
    text : str
        The text to clean
    keep_punctuation : bool
        Defines if punctuation should be kept
        
    Returns
    -------
    str
        The cleaned text
    """
    
    # remove any html tags (< /br> often found)
    text = re.sub(RE_TAGS, " ", text)
    
    if keep_punctuation:
        # keep only ASCII + European Chars and whitespace, no digits, keep punctuation
        text = re.sub(RE_ASCII_PUNCTUATION, " ", text)
        # convert all whitespaces (tabs etc.) to single wspace, keep punctuation
        text = re.sub(RE_SINGLECHAR_PUNCTUATION, " ", text)
    else:
        # keep only ASCII + European Chars and whitespace, no digits, no punctuation
        text = re.sub(RE_ASCII, " ", text)
        # convert all whitespaces (tabs etc.) to single wspace
        text = re.sub(RE_SINGLECHAR, " ", text)     
    
    text = re.sub(RE_WSPACE, " ", text)  
    return text

def clean_text_df(df, field_read="text_original", field_write="text_clean", keep_punctuation=False):
    """Cleans a column of text by calling clean_text (primarily meant for parallel processing)

    Parameters
    ----------
    df : dataframe
        The dataframe
    field_read : str
        The column name to read from (default is text_original)
    field_write : str
        The column name to write to (default is text_clean)
    keep_punctuation : bool
        Defines if punctuation should be kept
        
    Returns
    -------
    dataframe
        The dataframe with the cleaned text
    """
        
    # do not grow the dataframe directly - see https://stackoverflow.com/a/56746204
    series = df[field_read].map(
        lambda x: clean_text(x, keep_punctuation) if isinstance(x, str) or is_iterable(x) else list()
    )
    
    return series.to_frame(field_write)
    
from collections import Counter
import nltk
from matplotlib import pyplot as plt

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
        
    counter = Counter()

    # see https://stackoverflow.com/a/17071908
    #_ = df[field_read].apply(lambda x: counter.update(nltk.ngrams(x, n)))
    #_ = df[field_read].apply(lambda x: counter.update([" ".join(grams) for grams in nltk.ngrams(x, n)]))
    _ = df[field_read].apply(lambda x: counter.update([" ".join(x[i:i+n]) for i in range(len(x)-n+1)]))
    
    return counter
