
def get_train_test_split(params, data):
    """Performs a train/test split based on the provided params

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    data: dataframe
        The data
        
    Returns
    -------
    dataframe
        A tuple with the train/test data
    """
    
    from sklearn.model_selection import train_test_split

    classification_type = get_classification_type(params, data)
    verbose = params.get("verbose", False)
    split_size = params.get("train_test_split_size", 0.2)
    y_column_name = params.get("y_column_name", "label")

    if classification_type == "multi-label":
        data_train, data_test = multilabel_train_test_split(data, test_size=split_size, shuffle=True, random_state=42, stratify=data[y_column_name])
    else:
        data_train, data_test = train_test_split(data, test_size=split_size, shuffle=True, random_state=42, stratify=data[y_column_name])

    if verbose:
        print(len(data_train), 'train examples')
        print(len(data_test), 'test examples')
        
    return (data_train, data_test)


def multilabel_train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    """
    Train test split for multilabel classification. Uses the algorithm from: 
    'Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-Label Data'.

    Parameters
    ----------
    *arrays:
        sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    test_size:
        float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
    train_size:
        float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.
    random_state:
        int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
    shuffle:
        bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
    stratify:
        array-like, default=None
        If not None, data is split in a stratified fashion, using this as the class labels.
        
    """

    # see https://github.com/scikit-multilearn/scikit-multilearn/issues/202#issuecomment-1052868514

    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    from sklearn.utils import indexable, _safe_indexing
    from sklearn.utils.validation import _num_samples
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection._split import _validate_shuffle_split
    from itertools import chain
    from sklearn.preprocessing import MultiLabelBinarizer
    
    if stratify is None:
        return train_test_split(*arrays, test_size=test_size, train_size=train_size,
                                random_state=random_state, shuffle=shuffle, stratify=None)
    
    assert shuffle, "Stratified train/test split is not implemented for shuffle=False"
    
    n_arrays = len(arrays)
    arrays = indexable(*arrays)
    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )
    mlb = MultiLabelBinarizer()
    stratify = mlb.fit_transform(stratify)
    cv = MultilabelStratifiedShuffleSplit(test_size=n_test, train_size=n_train, random_state=random_state)
    train, test = next(cv.split(X=arrays[0], y=stratify))

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    ) 

    
def get_classification_type(params, data):
    """Determines the classification type based on what the user defined or inferred by the labels

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    data: dataframe
        The data
    """
    import numpy as np
    from pandas.api.types import is_list_like
    
    verbose = params.get("verbose", False)
    classification_type = params.get("classification_type", None)
    y_column_name = params.get("y_column_name", "label")
    
    if classification_type is None:
        if is_list_like(data[y_column_name].iloc[0]):
            classification_type = "multi-label"
        elif len(data[y_column_name].unique()) > 2:
            classification_type = "multi-class"
        else:
            classification_type = "binary"
        
        if verbose:
            print("Inferred classification type:", classification_type)
            
    return classification_type
    

def get_classification_type_and_set(params, data):
    """Determines the classification type based on what the user defined or inferred by the labels

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    data: dataframe
        The data
    """
    
    from fhnw.nlp.utils.params import get_classification_type
    
    classification_type =  get_classification_type(params, data)
    params["classification_type"] = classification_type
    return classification_type


def get_label_binarize(params, data):
    """Creates an initialized LabelBinarizer

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    data: dataframe
        The data
    """
    
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.preprocessing import MultiLabelBinarizer
    from fhnw.nlp.utils.params import get_classification_type
    
    y_column_name = params.get("y_column_name", "label")
    classification_type = get_classification_type(params, data)
    
    if classification_type == "multi-label":
        label_binarizer = MultiLabelBinarizer()
        _ = label_binarizer.fit(data[y_column_name])
    else: 
        label_binarizer = LabelBinarizer()
        _ = label_binarizer.fit(data[y_column_name])
    
    return label_binarizer


def create_label_binarizer_and_set(params, data):
    """Creates an initialized LabelBinarizer

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    data: dataframe
        The data
    """
    
    from fhnw.nlp.utils.params import get_classification_type_and_set
    from fhnw.nlp.utils.params import get_label_binarize

    computed_objects_column_name = params.get("computed_objects_column_name", "computed_objects")
    _ = get_classification_type_and_set(params, data)
    params.setdefault(computed_objects_column_name, {})["label_binarizer"] = get_label_binarize(params, data)

def compute_binarized_labels(params, data):
    """Binarizes the labels

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    data: dataframe
        The data
        
    Returns
    -------
    list
        The binarized labels
    """
    
    y_column_name = params.get("y_column_name", "label")
    computed_objects_column_name = params.get("computed_objects_column_name", "computed_objects")
    label_binarizer = params[computed_objects_column_name]["label_binarizer"]
    
    y = label_binarizer.transform(data[y_column_name])
    output_classes = len(label_binarizer.classes_)
    if output_classes <= 2:
        y = y.flatten()

    return y
    
    
def dataframe_to_dataset(params, data, training = True):
    """Converts a dataframe into a dataset

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    data: dataframe
        The data
    training: bool
        Indicates if it is for training or inference
        
    Returns
    -------
    dataset
        The dataset
    """
    
    import tensorflow as tf
    
    X_column_name = params.get("X_column_name", "text_clean")
    
    if training:
        y_column_name = params.get("y_column_name", "label")
        computed_objects_column_name = params.get("computed_objects_column_name", "computed_objects")
        label_binarizer = params.setdefault(computed_objects_column_name, {})["label_binarizer"]
        
        y = data[y_column_name]
        y = label_binarizer.transform(y)
        output_classes = len(label_binarizer.classes_)
        if output_classes <= 2:
            y = y.flatten()
            
        ds = tf.data.Dataset.from_tensor_slices((data[X_column_name].values, y))
    else:
        ds = tf.data.Dataset.from_tensor_slices((data[X_column_name].values))
        
    return ds


def build_preprocessed_dataset(params, dataset, training = False, preprocessor_time_intensive = None, preprocessor_memory_intensive = None):
    """Builds a preprocessed data pipeline optimized for GPUs 

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    dataset: tf.data.Dataset
        The Dataset
    training: bool
        Indicates if it is for training or inference
    preprocessor_time_intensive: function
        Time intensive preprocessing transformations
    preprocessor_memory_intensive: function
        Memory intensive preprocessing transformations
    """
    
    import tensorflow as tf
    
    batch_size = params.get("batch_size", 64)
    n_samples = params.get("n_samples")

    dataset_preprocessed = dataset
    # vectorize later transformations through batching
    dataset_preprocessed = dataset_preprocessed.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    # parallelize (time intensive) transformations
    if preprocessor_time_intensive is not None:
        dataset_preprocessed = dataset_preprocessed.map(preprocessor_time_intensive, num_parallel_calls=tf.data.AUTOTUNE)
    # cache preprocessed data
    dataset_preprocessed = dataset_preprocessed.cache()
    # shuffle data (e.g. to improve training)
    if training:
        dataset_preprocessed = dataset_preprocessed.shuffle(buffer_size=n_samples)
    # parallelize (memory intensive) transformations 
    if preprocessor_memory_intensive is not None:
        dataset_preprocessed = dataset_preprocessed.map(preprocessor_memory_intensive, num_parallel_calls=tf.data.AUTOTUNE)
    # prefetch data to overlap producer (e.g. preprocessing text data on CPU) and consumer (training tensor data on GPU)
    dataset_preprocessed = dataset_preprocessed.prefetch(tf.data.AUTOTUNE)
    
    return dataset_preprocessed

    
def extract_vocabulary_and_set(params, data):
    """Extracts the vocabulary and puts it into the params dictionary

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    data: dataframe
        The data
    """
    
    verbose = params.get("verbose", False)
    tokenized_column = params.get("tokenized_column", "token_clean")
    sequence_length_percentil_cutoff = params.get("sequence_length_percentil_cutoff", 0.98)
    sequence_length_max = params.get("sequence_length_max", 768)
    computed_objects_column_name = params.get("computed_objects_column_name", "computed_objects")
    X = data[tokenized_column]

    vocabulary = set()
    _ = X.apply(lambda x: vocabulary.update(x))

    lengths = X.apply(len)
    max_sequence_length = int(lengths.quantile(1.0))
    percentil_sequence_length = int(lengths.quantile(0.98))
    median_sequence_length = int(lengths.quantile(0.5))
    embedding_input_sequence_length = min(sequence_length_max, percentil_sequence_length)
    
    if verbose:
        print("Median sequence length:", median_sequence_length)
        print("Percentil (", sequence_length_percentil_cutoff, ") cutoff sequence length: ", percentil_sequence_length, sep='')
        print("Max sequence length:", max_sequence_length)
        print("Used embedding sequence length:", embedding_input_sequence_length)

    params.setdefault(computed_objects_column_name, {})["vocabulary"] = vocabulary
    params["embedding_input_sequence_length"] = embedding_input_sequence_length


def extract_text_vectorization_and_set(params):
    """Creates the TextVectorization layer and a vocabulary iterator and puts them into the params dictionary

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    """
    
    # see https://towardsdatascience.com/you-should-try-the-new-tensorflows-textvectorization-layer-a80b3c6b00ee
    try:
        # for newer tf versions use
        from tensorflow.keras.layers import TextVectorization
    except ImportError:
        from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

    verbose = params.get("verbose", False)
    output_sequence_length = params.get("output_sequence_length", None)
    computed_objects_column_name = params.get("computed_objects_column_name", "computed_objects")
    vocabulary = params.setdefault(computed_objects_column_name, {})["vocabulary"]
    
    vectorize_layer = TextVectorization(
        output_mode='int',
        output_sequence_length=output_sequence_length,
        vocabulary=list(vocabulary),
        name="text_vectorization"
    )
    
    params.setdefault(computed_objects_column_name, {})["vocabulary_iterator"] = vectorize_layer.get_vocabulary()
    params.setdefault(computed_objects_column_name, {})["vectorize_layer"] = vectorize_layer

    if verbose:
        print("Vocabulary length:", vectorize_layer.vocabulary_size())


def install_dependencies(params):
    """Installes the necessary dependencies for the given params setting

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    """
    
    import sys
    from fhnw.nlp.utils.system import install
    
    embedding_type = params["embedding_type"]
    
    if "fasttext" in embedding_type and "fasttext" not in sys.modules.keys():
        install("fasttext")
    if "word2vec" in embedding_type and "gensim" not in sys.modules.keys():
        install("gensim")
    if "spacy" in embedding_type and "spacy" not in sys.modules.keys():
        install("spacy")
        #install("click", "7.1.1")
        #install("Exit")
    if "tensorflow" in embedding_type and "tensorflow_hub" not in sys.modules.keys():
        install("tensorflow_hub")
    if "bytepair" in embedding_type and "bpemb" not in sys.modules.keys():
        install("bpemb")


def get_embedder_fasttext(params):
    """Provides the fasttext embedder

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    """
        
    import fasttext
    import fasttext.util

    embedding_dim = params["embedding_dim"]
    model_name = params["embedding_fasttext_model"]
    split = model_name.split(".")
    model_lang = split[1]
    model_dim = int(split[2])
    
    try:
        ft = fasttext.load_model(model_name)
    except ValueError:
        fasttext.util.download_model(model_lang, if_exists='ignore')
        ft = fasttext.load_model(model_name)
    
    if embedding_dim < model_dim:
        fasttext.util.reduce_model(ft, embedding_dim)
    
    def fasttext_embedder(word):
        return ft.get_word_vector(word)
    
    return fasttext_embedder


def get_embedder_byte_pair(params):
    """Provides the byte pair embedder

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    """
        
    from bpemb import BPEmb
    import numpy as np

    embedding_lang = params.get("embedding_lang", "de")
    
    bpemb = BPEmb(lang=embedding_lang)
    
    def byte_pair_embedder(word, order=2):
        vec = bpemb.embed(word).sum(axis=0)
        l2 = np.atleast_1d(np.linalg.norm(vec, order))
        l2[l2==0] = 1
        #return vec / np.expand_dims(l2, axis)
        return vec / l2
    
    return byte_pair_embedder

def get_embedder_word2vec(params):
    """Provides the word2vec embedder

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    """
        
    import os
    import gensim
    
    model_url = params["embedding_word2vec_model_url"]
    model_path = "models/word2vec/"+os.path.basename(model_url)
    
    if not os.path.exists(model_path):
        download(url=model_url, path = model_path)
        
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    
    def word2vec_embedder(word):
        try:
            return model[word]
        except KeyError:
            return None
    
    return word2vec_embedder
   
    
def get_embedder_spacy(params):
    """Provides the spacy embedder

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    """
        
    import spacy
    
    model_name = params["embedding_spacy_model"]
    
    try:
        nlp = spacy.load(model_name)
    except OSError:
        from spacy.cli import download
        
        download(model_name)
        nlp = spacy.load(model_name)
    
    def spacy_embedder(word):
        return nlp(word)[0].vector
    
    return spacy_embedder


def get_embedder_tensorflow_hub(params):
    """Provides the tensorflow embedder

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    """
    
    import tensorflow_hub as hub
    
    embedding_url = params["embedding_tensorflow_hub_url"]
    
    embed = hub.load(embedding_url)
    
    def tensorflow_hub_embedder(word):
        return embed([word])[0].numpy()
    
    return tensorflow_hub_embedder


def get_embedder(params):
    """Provides the embedder based on the params

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    """
    
    embedding_type = params["embedding_type"]
    if embedding_type == "fasttext":
        return get_embedder_fasttext(params)
    if embedding_type == "word2vec":
        return get_embedder_word2vec(params)
    if embedding_type == "spacy":
        return get_embedder_spacy(params)
    if embedding_type == "tensorflow_hub":
        return get_embedder_tensorflow_hub(params)
    if embedding_type == "bytepair":
        return get_embedder_byte_pair(params)
    else:
        raise TypeError("Unknown embedding_type "+ embedding_type)
    

def calculate_embedding_matrix(params, embedder):
    """Creates the embedding matrix

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    embedder: function
        The function to get the embedding for a word
    """    

    import numpy as np

    computed_objects_column_name = params.get("computed_objects_column_name", "computed_objects")
    vocabulary_iterator = params.setdefault(computed_objects_column_name, {})["vocabulary_iterator"]
    tmp_embedding = embedder("haus")
    
    if (tmp_embedding is not None) and len(tmp_embedding) > 0:
        embedding_dim = len(tmp_embedding)
    else:
        embedding_dim = params["embedding_dim"]
    voc_size = len(vocabulary_iterator)
    words_not_found = set()
    embedding_matrix = np.zeros((voc_size, embedding_dim))

    for idx, word in enumerate(vocabulary_iterator):
        embedding_vector = embedder(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0 and not np.all(embedding_vector==0):
            # words not found in embedding index will be all-zeros.
            embedding_matrix[idx] = embedding_vector
        else:
            words_not_found.add(word)

    if params["verbose"]:
        print("Embedding type:", params.get("embedding_type"))
        print("Number of null word embeddings:", np.sum(np.sum(embedding_matrix, axis=1) == 0))
        nr_words_not_found = len(words_not_found)
        print("Words not found in total:", len(words_not_found))
        if nr_words_not_found > 0:
            import random
            
            nr_sample = min(20, len(words_not_found))
            print("Words without embedding (", nr_sample, "/", nr_words_not_found, "): ", random.sample(list(words_not_found), nr_sample), sep='')
        
    return embedding_matrix


def extract_embedding_layer_and_set(params):
    """Creates the Embedding layer and puts it into the params dictionary

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    """
    
    import tensorflow as tf
    from tensorflow import keras
    from fhnw.nlp.utils.params import install_dependencies
    
    computed_objects_column_name = params.get("computed_objects_column_name", "computed_objects")
    
    install_dependencies(params)
    
    embedding_type = params.get("embedding_type")
    if embedding_type is not None:
        if embedding_type == "tensorflow_hub_layer":
            import tensorflow_hub as hub
            
            embedding_url = params["embedding_tensorflow_hub_url"]
            embedding_layer = hub.KerasLayer(embedding_url, input_shape=[], dtype=tf.string, trainable=params["embedding_trainable"], name="hub_embedding")
        else:
            embedder = get_embedder(params)       
            embedding_matrix = calculate_embedding_matrix(params, embedder)       
            embedding_layer = keras.layers.Embedding(
                                          embedding_matrix.shape[0], 
                                          embedding_matrix.shape[1], 
                                          weights=[embedding_matrix],
                                          input_length=params["embedding_input_sequence_length"],
                                          trainable=params.get("embedding_trainable", False),
                                          mask_zero = params.get("embedding_mask_zero", True),
                                          name="embedding"
                                         )
    else:
        embedding_layer = keras.layers.Embedding(
                                                 len(params["vocabulary_iterator"]), 
                                                 params["embedding_dim"], 
                                                 mask_zero = params["embedding_mask_zero"], 
                                                 trainable=True)
        
    params.setdefault(computed_objects_column_name, {})["embedding_layer"] = embedding_layer
    

def compile_model(params, model):
    """Compiles the model based on the provided params 

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    model: model
        The keras model
    """
    
    from tensorflow import keras
        
    optimizer_learning_rate = params.get("learning_rate", 0.01)
    optimizer_learning_rate_decay = params.get("optimizer_learning_rate_decay", None)
    model_metric = get_model_metric(params)
    model_loss_function = get_loss_function(params)

    adam = keras.optimizers.Adam(learning_rate=optimizer_learning_rate)
    if optimizer_learning_rate_decay is not None:
        adam = keras.optimizers.Adam(learning_rate=optimizer_learning_rate, decay=optimizer_learning_rate_decay)

    #model.compile(loss=model_loss_function, optimizer=adam, metrics=model_metric, jit_compile=True)
    model.compile(loss=model_loss_function, optimizer=adam, metrics=model_metric)


def create_text_preprocessor(params, training = False):
    """Creates a text preprocessor based on the provided params 

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    training: bool
        Indicates if it is for training or inference
    """
    
    computed_objects_column_name = params.get("computed_objects_column_name", "computed_objects")
    vectorize_layer = params[computed_objects_column_name]["vectorize_layer"]
    
    if training:
        # preprocessing function for text and label data (e.g. vectorize training data (text and label))
        text_preprocessor = lambda text, label: (vectorize_layer(text), label)
    else:
        # preprocessing function for text data (e.g. vectorize inference data (text only))
        text_preprocessor = lambda text: vectorize_layer(text)

    return text_preprocessor


def build_model_cnn(params):
    """Builds a cnn classifier based on the provided params 

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    """
        
    # https://towardsdatascience.com/pretrained-word-embeddings-using-spacy-and-keras-textvectorization-ef75ecd56360
    # https://colab.research.google.com/drive/1RvCnR7h0_l4Ekn5vINWToI9TNJdpUZB3#scrollTo=fAnI0YwfvXdG
    
    # Regularization
    # https://medium.com/intelligentmachines/convolutional-neural-network-and-regularization-techniques-with-tensorflow-and-keras-5a09e6e65dc7
    
    import tensorflow as tf
    from tensorflow import keras
    
    cnn_num_conv_pooling_layers = params.get("cnn_num_conv_pooling_layers", 2)
    cnn_conv_num_filters = params.get("cnn_conv_num_filters", 128)
    cnn_conv_kernel_size = params.get("cnn_conv_kernel_size", 7)
    cnn_conv_activation_function = params.get("cnn_conv_activation_function", "relu")
    cnn_conv_strides = params.get("cnn_conv_strides", 1)
    cnn_conv_padding = params.get("cnn_conv_padding", "valid")
    cnn_max_pool_size = params.get("cnn_max_pool_size", 2)
    cnn_max_pool_strides = params.get("cnn_max_pool_strides", None)
    cnn_max_pool_padding = params.get("cnn_max_pool_padding", "valid")
    cnn_global_max_pool_dropout = params.get("cnn_global_max_pool_dropout", 0.5)
    cnn_dense_units = params.get("cnn_dense_units", 128)
    cnn_dense_activation_function = params.get("cnn_dense_activation_function", "relu")
    cnn_dense_kernel_regularizer_l1 = params.get("cnn_dense_kernel_regularizer_l1", None)
    cnn_dense_kernel_regularizer_l2 = params.get("cnn_dense_kernel_regularizer_l2", None)
    cnn_output_dropout = params.get("cnn_output_dropout", 0.5)
    
    classification_type = params.get("classification_type", "binary")
    
    X_column_name = params.get("X_column_name", "text_clean")
    computed_objects_column_name = params.get("computed_objects_column_name", "computed_objects")
    vectorize_layer = params[computed_objects_column_name]["vectorize_layer"]
    embedding_layer = params[computed_objects_column_name]["embedding_layer"]
    label_binarizer = params[computed_objects_column_name]["label_binarizer"]
    output_classes = len(label_binarizer.classes_)
    output_classes = output_classes if output_classes > 2 else 1
    
    # Build a separate model for training
    model_train = keras.Sequential(name="cnn_train")
    # The input for the training model is already processes, i.e. vectorized.
    # Offloading this step and prefetching the data provides speedup during training 
    # After vectorization we have a tensor of shape (batch_size, output_sequence_length) containing vocab indices.
    model_train.add(keras.Input(shape=(None,), dtype=tf.int64, name="preprocessed_input"))
    # Next, we add a layer to map those vocab indices into a space of dimensionality 'embedding_dim'. 
    model_train.add(embedding_layer)
    
    for layer in range(cnn_num_conv_pooling_layers):
        model_train.add(keras.layers.Conv1D(cnn_conv_num_filters, cnn_conv_kernel_size, activation=cnn_conv_activation_function, strides=cnn_conv_strides, padding=cnn_conv_padding, name="conv_"+str(layer)))
        
        if layer + 1 < cnn_num_conv_pooling_layers:
            model_train.add(keras.layers.MaxPooling1D(pool_size=cnn_max_pool_size, strides=cnn_max_pool_strides, padding=cnn_max_pool_padding, name="max_pool_"+str(layer)))
        else:
            model_train.add(keras.layers.GlobalMaxPooling1D(name="global_max_pool_"+str(layer)))
    
    
    if cnn_global_max_pool_dropout is not None and cnn_global_max_pool_dropout > 0 and cnn_num_conv_pooling_layers > 0:
        model_train.add(keras.layers.Dropout(cnn_global_max_pool_dropout, name="global_max_pool_dropout"))
    

    kernel_regularizer = None
    if cnn_dense_kernel_regularizer_l1 is not None and cnn_dense_kernel_regularizer_l1 > 0 and cnn_dense_kernel_regularizer_l2 is not None and cnn_dense_kernel_regularizer_l2 > 0:
        kernel_regularizer = regularizers.l1_l2(l1=cnn_dense_kernel_regularizer_l1, l2=cnn_dense_kernel_regularizer_l2)
    elif cnn_dense_kernel_regularizer_l1 is not None and cnn_dense_kernel_regularizer_l1 > 0:
        kernel_regularizer = regularizers.l1(cnn_dense_kernel_regularizer_l1)
    elif cnn_dense_kernel_regularizer_l2 is not None and cnn_dense_kernel_regularizer_l2 > 0:
        kernel_regularizer = regularizers.l2(cnn_dense_kernel_regularizer_l2)
    model_train.add(keras.layers.Dense(cnn_dense_units, activation=cnn_dense_activation_function, kernel_regularizer=kernel_regularizer, name="dense"))
    
    
    if cnn_output_dropout is not None and cnn_output_dropout > 0:
        model_train.add(keras.layers.Dropout(cnn_output_dropout, name="dense_dropout"))

    
    if classification_type == "binary":
        output_activation = "sigmoid"
    elif classification_type == "multi-class":
        output_activation = "softmax"
    elif classification_type == "multi-label":
        output_activation = "sigmoid"
    else:
        raise TypeError("Unknown classification_type "+classification_type)

    model_train.add(keras.layers.Dense(output_classes, activation=output_activation, name="prediction"))
    
    
    # Next we build the inference model that also contains the preprocessing (all in one)
    model_inference = keras.Sequential(name="cnn_inference")
    # A text input layer
    model_inference.add(keras.layers.InputLayer(input_shape=(1,), dtype=tf.string, name=X_column_name))
    # Then we vectorize the text.
    # After this layer, we have a tensor of shape (batch_size, output_sequence_length) containing vocab indices.
    model_inference.add(vectorize_layer)
    # Next we just re-use the training model 
    model_inference.add(model_train)
    
    
    return model_train, model_inference
    
    
def build_model_rnn(params):
    """Builds a rnn classifier based on the provided params 

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    """
    
    import tensorflow as tf
    from tensorflow import keras
    
    rnn_activation_function = params.get("rnn_activation_function", "relu")
    rnn_output_dropout = params.get("rnn_output_dropout", 0.5)
    rnn_dropout = params.get("rnn_dropout", 0.2)
    # for values > 0 it will not use cuDNN kernels
    rnn_recurrent_dropout = params.get("rnn_recurrent_dropout", 0.0)
    rnn_units = params.get("rnn_units", 32)
    rnn_num_layers = params.get("rnn_num_layers", 1)
    rnn_bidirectional = params.get("rnn_bidirectional", True)
    rnn_type = params.get("rnn_type", "LSTM")
    rnn_global_max_pooling = params.get("rnn_global_max_pooling", False)

    classification_type = params.get("classification_type", "binary") 
    
    X_column_name = params.get("X_column_name", "text_clean")
    computed_objects_column_name = params.get("computed_objects_column_name", "computed_objects")
    vectorize_layer = params[computed_objects_column_name]["vectorize_layer"]
    embedding_layer = params[computed_objects_column_name]["embedding_layer"]
    label_binarizer = params[computed_objects_column_name]["label_binarizer"]
    output_classes = len(label_binarizer.classes_)
    output_classes = output_classes if output_classes > 2 else 1
    
    model_train = keras.Sequential(name="rnn_train")
    # The input for the training model is already processes, i.e. vectorized.
    # Offloading this step and prefetching the data provides speedup during training 
    # After vectorization we have a tensor of shape (batch_size, output_sequence_length) containing vocab indices.
    model_train.add(keras.Input(shape=(None,), dtype=tf.int64, name="preprocessed_input"))
    # Next, we add a layer to map those vocab indices into a space of dimensionality 'embedding_dim'. 
    model_train.add(embedding_layer)
    
    for layer in range(rnn_num_layers):
        return_sequences = layer + 1 < rnn_num_layers or rnn_global_max_pooling
        
        if rnn_type == "LSTM":
            layer = keras.layers.LSTM(rnn_units, return_sequences=return_sequences, dropout=rnn_dropout, recurrent_dropout=rnn_recurrent_dropout)
        elif rnn_type == "GRU":
            layer = keras.layers.GRU(rnn_units, return_sequences=return_sequences, dropout=rnn_dropout, recurrent_dropout=rnn_recurrent_dropout)
        elif rnn_type == "RNN":
            layer = keras.layers.RNN(rnn_units, return_sequences=return_sequences)
        else:
            raise TypeError("Unknown rnn_type "+rnn_type)
        
        
        if rnn_bidirectional:
            layer = keras.layers.Bidirectional(layer)
         
        model_train.add(layer)
        
        rnn_units = int(rnn_units / 2)
    
    if rnn_global_max_pooling:
        model_train.add(keras.layers.GlobalMaxPool1D(name="global_max_pool"))
    
    model_train.add(keras.layers.Dense(rnn_units, activation=rnn_activation_function))
    
    if rnn_output_dropout is not None and rnn_output_dropout > 0:
        model_train.add(keras.layers.Dropout(rnn_output_dropout, name="dense_dropout"))
        
    if classification_type == "binary":
        output_activation = "sigmoid"
    elif classification_type == "multi-class":
        output_activation = "softmax"
    elif classification_type == "multi-label":
        output_activation = "sigmoid"
    else:
        raise TypeError("Unknown classification_type "+classification_type)

    model_train.add(keras.layers.Dense(output_classes, activation=output_activation, name="prediction"))
    
    
    # Next we build the inference model that also contains the preprocessing (all in one)
    model_inference = keras.Sequential(name="rnn_inference")
    # A text input layer
    model_inference.add(keras.layers.InputLayer(input_shape=(1,), dtype=tf.string, name=X_column_name))
    # Then we vectorize the text.
    # After this layer, we have a tensor of shape (batch_size, output_sequence_length) containing vocab indices.
    model_inference.add(vectorize_layer)
    # Next we just re-use the training model 
    model_inference.add(model_train)
    
    
    return model_train, model_inference


def get_loss_function(params):
    """Decides upon the loss function to use based on the provided params

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
        
    Returns
    -------
    str
        The name of the loss function (or a callable)
    """
        
    classification_type = params.get("classification_type", "binary")
    
    if classification_type == "binary":
        model_loss_function = "binary_crossentropy"
    elif classification_type == "multi-class":
        model_loss_function = "categorical_crossentropy"
    elif classification_type == "multi-label":
        model_loss_function = "binary_crossentropy"
    else:
        raise TypeError("Unknown classification_type "+classification_type)
    
    return model_loss_function


def get_model_metric(params):
    """Provides the metric based on the provided params

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
        
    Returns
    -------
    list
        The list of metrics to use
    """
    
    return params.get("model_metric", ["accuracy"])


def compile_model(params, model):
    """Compiles the model based on the provided params 

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    model: model
        The keras model
    """
    
    from tensorflow import keras
    from fhnw.nlp.utils.params import get_loss_function
    from fhnw.nlp.utils.params import get_model_metric
        
    optimizer_learning_rate = params.get("learning_rate", 0.01)
    optimizer_learning_rate_decay = params.get("optimizer_learning_rate_decay", None)
    model_metric = get_model_metric(params)
    model_loss_function = get_loss_function(params)

    adam = keras.optimizers.Adam(learning_rate=optimizer_learning_rate)
    if optimizer_learning_rate_decay is not None:
        adam = keras.optimizers.Adam(learning_rate=optimizer_learning_rate, decay=optimizer_learning_rate_decay)

    #model.compile(loss=model_loss_function, optimizer=adam, metrics=model_metric, jit_compile=True)
    model.compile(loss=model_loss_function, optimizer=adam, metrics=model_metric)
    
    
def re_compile_model(params, model):
    """Re-compiles the model based on the provided params and the existing optimizer 

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    model: model
        The keras model
    """
    
    from tensorflow import keras
        
    # use a low learning rate for fine tuning
    optimizer_learning_rate = params.get("learning_rate", 1e-5)
    optimizer_learning_rate_decay = params.get("optimizer_learning_rate_decay", None)
    model_metric = get_model_metric(params)
    model_loss_function = get_loss_function(params)
    
    # keep existing internal parameters for further runs
    optimizer = model.optimizer
    optimizer.learning_rate.assign(optimizer_learning_rate)
    if optimizer_learning_rate_decay is not None:
    	optimizer.decay.assign(optimizer_learning_rate_decay)

    model.compile(loss=model_loss_function, optimizer=optimizer, metrics=model_metric)


def train_model(params, model, dataset_train, dataset_val):
    """Performs the model training 

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    model: model
        The keras model
    dataset_train: tf Dataset
        The dataset for training
    dataset_val; tf Dataset
        The dataset for validation
        
    Returns
    -------
    history
        The training history
    """
        
    import os
    import datetime
    from tensorflow import keras

    training_epochs = params.get("training_epochs", 5)
    
    #training_logdir = params.get("training_logdir", None)
    #if training_logdir is None:
    #    training_logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=0, profile_batch="10, 15")

    history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        #callbacks=[tensorboard_callback],
        epochs=training_epochs)
    
    return history
    

def predict_classification(params, data, model, preprocessor = None):
    """Predicts the classes 

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    data: dataframe
        The data
    model: model
        The keras model
    preprocessor: function
        Preprocessing transformations like text vectorization (if None this must be part of the model)
        
    Returns
    -------
    lists/arrays
        The true labels (ground truth), the predicted labels, and the prediction probabilities
    """
    
    import numpy as np
    from fhnw.nlp.utils.params import dataframe_to_dataset
    from fhnw.nlp.utils.params import build_preprocessed_dataset
        
    batch_size = 2 * params.get("batch_size", 64)
    y_column_name = params.get("y_column_name", "label")
    X_column_name = params.get("X_column_name", "text_clean")
    prediction_probability_threshold = params.get("prediction_probability_threshold", 0.5)
    computed_objects_column_name = params.get("computed_objects_column_name", "computed_objects")
    label_binarizer = params[computed_objects_column_name]["label_binarizer"]
    
    y = data[y_column_name]
        
    dataset = dataframe_to_dataset(params, data, False)
    X = build_preprocessed_dataset(params, dataset, False, preprocessor)
    
    y_pred_prob = model.predict(X, batch_size=batch_size)
    y_pred = label_binarizer.inverse_transform(y_pred_prob, threshold=prediction_probability_threshold)                                             
    
    params["labels"] = y
    params["labels_predicted"] = y_pred
    params["labels_predicted_probability"] = y_pred_prob
    
    return (y, y_pred, y_pred_prob)


def save_model(params, model, history):
    """Saves the model to disk

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    model: model
        The keras model
    history
        The training history
    """
    
    import os
    import datetime

    model_save_path = params.get("model_save_path", None)
    
    if model_save_path is None:
        model_type = params.get("model_type", "unknown")
            
        model_save_path = os.path.join("models", model_type, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), "model_"+ f"{history.history['val_accuracy'][len(history.history['val_accuracy']) - 1]:.5f}")
        os.makedirs(model_save_path, exist_ok=True)       

    model.save(model_save_path, save_format='tf')
