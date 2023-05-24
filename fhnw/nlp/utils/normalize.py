def tokenize(text, stopwords, word_splitter):
    """Tokenizes and lowercases a text and removes stopwords

    Parameters
    ----------
    text : str, iterable
        The text either as string or iterable of tokens (in this case tokenization is not applied)
    stopwords : set
        A set of stopword to remove from the tokens
    word_splitter: splitter
        The word splitter to use (callable e.g. compound_split to split compound words) or None to disable word splitting
        
    Returns
    -------
    list
        The tokenized text
    """
    from fhnw.nlp.utils.processing import is_iterable
    
    if isinstance(text, str):
        from fhnw.nlp.utils.defaults import default_tokenizer
        word_tokens = default_tokenizer()(text)
    elif is_iterable(text):
        word_tokens = text
    else:
        raise TypeError("Only string or iterable (e.g. list) is supported. Received a "+ str(type(text)))
        
    if word_splitter is not None:
        # with python 3.8
        #return [lower for word in word_tokens if word.lower() not in stopwords for split in word_splitter(word) if (lower := split.lower()) not in stopwords]
        return [split.lower() for word in word_tokens if word.lower() not in stopwords for split in word_splitter(word) if split.lower() not in stopwords]
    else:
        # with python 3.8
        #return [lower for word in word_tokens if (lower := word.lower()) not in stopwords]
        return [word.lower() for word in word_tokens if word.lower() not in stopwords]


def tokenize_stem(text, stopwords, word_splitter, stemmer):
    """Tokenizes, lowercases and stems a text and removes stopwords

    Parameters
    ----------
    text : str, iterable
        The text either as string or iterable of tokens (in this case tokenization is not applied)
    stopwords : set
        A set of stopword to remove from the tokens
    stemmer: callable
        The stemmer to use (callable e.g. SnowballStemmer)
    word_splitter: splitter
        The word splitter to use (callable e.g. compound_split to split compound words) or None to disable word splitting
        
    Returns
    -------
    list
        The tokenized and stemmed text
    """
    from typing import Callable
    from fhnw.nlp.utils.processing import is_iterable
        
    if isinstance(text, str):
        from fhnw.nlp.utils.defaults import default_tokenizer
        word_tokens = default_tokenizer()(text)
    elif is_iterable(text):
        word_tokens = text
    else:
        raise TypeError("Only string or iterable (e.g. list) is supported. Received a "+ str(type(text)))

    if not isinstance(stemmer, Callable):
        stemmer = stemmer.stem 

    if word_splitter is not None:
        # with python 3.8
        #return [stemmer(lower) for word in word_tokens if word.lower() not in stopwords for split in word_splitter(word) if (lower := split.lower()) not in stopwords]
        return [stemmer(split.lower()) for word in word_tokens if word.lower() not in stopwords for split in word_splitter(word) if split.lower() not in stopwords]
    else:
        # with python 3.8
        #return [stemmer(lower) for word in word_tokens if (lower := word.lower()) not in stopwords]
        return [stemmer(word.lower()) for word in word_tokens if word.lower() not in stopwords]


def tokenize_lemma(text, stopwords, lemmanizer, keep_ners=False):
    """Tokenizes, lowercases and lemmatizes a text and removes stopwords

    Parameters
    ----------
    text : str, iterable
        The text either as string or iterable of tokens (in this case tokenization is not applied)
    stopwords : set
        A set of stopword to remove from the tokens
    lemmanizer: spacy nlp pipeline
        The lemmanizer to use (must be spacy nlp pipeline)
    keep_ner: bool
        Defines if named entities (NERs) should be keept in one token
        
    Returns
    -------
    list
        The tokenized and lemmatized text
    """
    from fhnw.nlp.utils.processing import is_iterable
        
    if isinstance(text, str):
        text = text
    elif is_iterable(text):
        from fhnw.nlp.utils.text import join_tokens
        text = join_tokens(text, set())
    else:
        raise TypeError("Only string or iterable (e.g. list) is supported. Received a "+ str(type(text)))  
    
    if keep_ners:
        # HanoverTagger could be an alternative but takes longer 
        # see: https://textmining.wp.hs-hannover.de/Preprocessing.html#Lemmatisierung-und-Wortarterkennung
        doc = lemmanizer(text, disable=['tagger', 'parser'])

        tokens = list()
        ner_idx = 0
        tok_idx = 0

        # keep ner in one token
        while tok_idx < len(doc):
            if ner_idx >= len(doc.ents) or doc[tok_idx].idx < doc.ents[ner_idx].start_char:
                if doc[tok_idx].is_alpha and not doc[tok_idx].is_punct and doc[tok_idx].text.lower() not in stopwords and doc[tok_idx].lemma_.lower() not in stopwords:
                    #print("token ", doc[tok_idx].lemma_.lower())
                    tokens.append(doc[tok_idx].lemma_.lower())

                tok_idx += 1
            else:
                #print("ner ", doc.ents[ner_idx].lemma_.lower())
                tokens.append(doc.ents[ner_idx].lemma_.lower())

                tok_idx += 1
                while tok_idx < len(doc) and doc[tok_idx].idx < doc.ents[ner_idx].end_char:
                    tok_idx += 1

                ner_idx += 1

        return tokens
    else:
        doc = lemmanizer(text, disable=['tagger', 'parser', 'ner']) 
        return [tok.lemma_.lower() for tok in doc if tok.is_alpha and not tok.is_punct and tok.text.lower() not in stopwords and tok.lemma_.lower() not in stopwords]


def normalize(text, stopwords, word_splitter=None, stemmer=None, lemmanizer=None, lemma_with_ner=False):
    """Normalizes (e.g. tokenize and stem) and lowercases a text and removes stopwords

    Parameters
    ----------
    text : str, iterable
        The text either as string or iterable of tokens (in this case tokenization is not applied)
    stopwords : set
        A set of stopword to remove from the tokens
    word_splitter: callable
        The word splitter to use (callable e.g. compound_split to split compound words) or None to disable word splitting
    stemmer: callable
        The stemmer to use (callable e.g. SnowballStemmer) or None to disable stemming
    lemmanizer: spacy nlp pipeline
        The lemmanizer to use (must be spacy nlp pipeline) or None to disable lemmantization
    lemma_with_ner: bool
        Defines if named entities (NERs) should be keept in one token
        
    Returns
    -------
    list
        The normalized text
    """
        
    if lemmanizer is not None:
        return tokenize_lemma(text, stopwords, lemmanizer, keep_ners=lemma_with_ner)
    elif stemmer is not None:
        return tokenize_stem(text, stopwords, word_splitter, stemmer)
    else:
        return tokenize(text, stopwords, word_splitter)
