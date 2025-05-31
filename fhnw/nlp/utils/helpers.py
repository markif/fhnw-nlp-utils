def get_encoding(elements):
    """extracts the unique elements 

    Parameters
    ----------
    elements: list like
        The elements (and be 2D array like)
        
    Returns
    -------
    str
        The encoding (one-hot, index, label)
    """
    
    import numpy as np
    from pandas.api.types import is_list_like

    # check for multi-label
    if is_list_like(labels[0]):
        first_label = labels[0][0]
    else:
        first_label = labels[0]

    if isinstance(first_label, numbers.Number):
        if np.max(labels) <= 1:
            return "one-hot"
        else:
            return "index"
    else:
        return "label"
        

def get_unique_elements(elements, sort=True):
    """extracts the unique elements 

    Parameters
    ----------
    elements: list like
        The elements (and be 2D array like)
    sort: bool
        If the returned list should be sorted
        
    Returns
    -------
    list
        The unique elements
    """
    
    #import pandas as pd
    from pandas.api.types import is_list_like
    
    #unique_elements = pd.Series(elements)
    #
    #if is_list_like(elements[0]):
    #    unique_elements = unique_elements.explode()
    #
    #unique_elements = unique_elements.drop_duplicates().dropna().tolist()
    
    unique_elements = set()
    # check for multi-label
    list_like = is_list_like(elements[0])
    
    for element in elements:
        if list_like is True:
            unique_elements.update(element)
        else:
            unique_elements.add(element)
            
    unique_elements = list(unique_elements)
    
    if sort is True:
        unique_elements.sort()
    return unique_elements


def indices_to_one_hot(indices, num_labels, dtype=int):
    """Converts a list of indices array to a one-hot encoded matrix

    Parameters
    ----------
    indices: list
        The list containing array of indices
    num_labels: int
        The total number of labels
    dtype: type
        The type of the one-hot encoded elements (e.g. int, np.float32) 
        
    Returns
    -------
    ndarray
        The one-hot encoded matrix
    """
    # https://stackoverflow.com/questions/29034928/pandas-convert-a-column-of-list-to-dummies
    # https://stackoverflow.com/questions/56123419/how-to-cover-a-label-list-under-the-multi-label-classification-context-into-one
    import pandas as pd
    import numpy as np

    one_hot = pd.Series(indices)
    # ensure we consider all possible labels (indices might only contain [0,2,5] but we have 10 in total)
    # add a temporal rows with all possible labels
    one_hot[len(one_hot)] = [i for i in range(num_labels)]
    one_hot = one_hot.explode()   
    one_hot = pd.crosstab(one_hot.index, one_hot, dropna=False)
    if np.nan in one_hot.columns:
        # drop potential NaN column
        one_hot = one_hot.drop(columns=np.nan)
    # remove temporal num_classes row at end
    one_hot = one_hot.head(-1)
    #one_hot = one_hot.to_numpy()
    one_hot = one_hot.to_numpy(dtype=dtype)
    
    return one_hot


def labels_to_one_hot(labels, all_labels=None, dtype=int):
    """Converts a list of indices array to a one-hot encoded matrix

    Parameters
    ----------
    labels: list
        The list containing array of labels
    all_labels: list
        The all existing labels
    dtype: type
        The type of the one-hot encoded elements (e.g. int, np.float32) 
        
    Returns
    -------
    ndarray
        The one-hot encoded matrix
    """
    # https://stackoverflow.com/questions/29034928/pandas-convert-a-column-of-list-to-dummies
    # https://stackoverflow.com/questions/56123419/how-to-cover-a-label-list-under-the-multi-label-classification-context-into-one
    import pandas as pd
    import numpy as np

    one_hot = pd.Series(labels)
    
    if all_labels is None:
        all_labels = get_unique_elements(labels)

    # ensure we consider all possible labels (labels might only contain ["a","c","f"] but we have 10 in total)
    # add a temporal rows with all possible labels
    one_hot = pd.concat([pd.Series([all_labels]), one_hot]).reset_index(drop=True)
    one_hot = one_hot.explode()   
    one_hot = pd.crosstab(one_hot.index, one_hot, dropna=False)
    if np.nan in one_hot.columns:
        # drop potential NaN column
        one_hot = one_hot.drop(columns=np.nan)
    # remove temporal row at start
    one_hot = one_hot.tail(-1)
    #one_hot = one_hot.to_numpy()
    one_hot = one_hot.to_numpy(dtype=dtype)
    
    return one_hot    
    

def indices_to_labels(indices, index2label):
    """Converts indices into labels

    Parameters
    ----------
    labels: 2D array
        The labels
    index2label: dict
        The mapping from the index to the label
        
    Returns
    -------
    2D array
        The labels
    """
    
    import numpy as np
    from pandas.api.types import is_list_like

    # check for multi-label
    if is_list_like(indices[0]):
        return [[index2label[idx] for idx in idcs] for idcs in indices]
    else:
        return [index2label[idx] for idx in indices]


def labels_to_indices(labels, label2index):
    """Converts labels into indices

    Parameters
    ----------
    labels: 2D array
        The labels
    label2index: dict
        The mapping from the label to the index
        
    Returns
    -------
    2D array
        The indices
    """
    
    import numpy as np
    from pandas.api.types import is_list_like

    # check for multi-label
    if is_list_like(labels[0]):
        return [[label2index[label] for label in lbls] for lbls in labels]
    else:
        return [label2index[label] for label in labels]


def one_hot_to_indices(one_hot, classification_type = "binary"):
    """Converts one-hot into indices

    Parameters
    ----------
    one_hot: 2D array
        The one-hot encoded predictions
    classification_type: str
        The type of the classification (binary, multi-class, multi-label)
        
    Returns
    -------
    2D array
        The indices
    """
    
    import numpy as np
    
    if classification_type == "multi-label":
        classes = np.arange(len(one_hot[0]))

        return [classes.compress(indicators) for indicators in one_hot]
        
        #lb = MultiLabelBinarizer()
        #lb.fit([np.arange(len(one_hot[0]))])
        #provides tuples
        #return lb.inverse_transform(one_hot)
        # see https://stackoverflow.com/a/53859634
        # does not consider "no prediction" i.e. all elements 0
        #indices = np.argwhere(one_hot == 1)
        #return np.split(indices[:,1], np.unique(indices[:,0], return_index = True)[1])[1:]
        
    else:        
        return np.argmax(one_hot, axis=-1)    
    
    
def predict_from_logits(logits, num_labels, classification_type = "binary", encoding = "index", index2label = {}, prediction_probability_thresholds_by_index = {}):
    """Converts logits into predictions

    Parameters
    ----------
    logits: 2D array
        The logit values
    num_labels: int
        The total number of labels
    classification_type: str
        The type of the classification (binary, multi-class, multi-label)
    encoding: str
        The encoding of the prediction (index, one-hot, label, probs)
    index2label: dict
        The mapping from the one-hot index to the label
    prediction_probability_thresholds_by_index: dict
        The probability thresholds by index when the sigmoid value of a multi-label class is considered as predicted
        
    Returns
    -------
    2D array
        The predictions encoded as defined by encoding
    """
    
    if classification_type == "multi-label":
        import numpy as np
        
        # sigmoid
        probs = 1/(1 + np.exp(-logits))
        #import tensorflow as tf
        #probs = tf.keras.activations.sigmoid(tf.convert_to_tensor(logits))
        #import torch
        #sigmoid = torch.nn.Sigmoid()
        #probs = sigmoid(torch.Tensor(logits))        
    else:
        from sklearn.utils.extmath import softmax
        
        probs = softmax(logits) 
        #import tensorflow as tf
        #probs = tf.keras.activations.softmax(tf.convert_to_tensor(logits))
        #import torch
        #softmax = torch.nn.Softmax()
        #probs = softmax(torch.Tensor(logits))
    
    if encoding == "probs":
        return probs
    else:
        return predict_from_probs(probs, num_labels, classification_type, encoding, index2label, prediction_probability_thresholds_by_index)


def predict_from_probs(probs, num_labels, classification_type = "binary", encoding = "index", index2label = {}, prediction_probability_thresholds_by_index = {}):
    """Converts probabilities into predictions

    Parameters
    ----------
    logits: 2D array
        The logit values
    num_labels: int
        The total number of labels
    classification_type: str
        The type of the classification (binary, multi-class, multi-label)
    encoding: str
        The encoding of the prediction (index, one-hot, label, probs)
    index2label: dict
        The mapping from the one-hot index to the label
    prediction_probability_thresholds_by_index: dict
        The probability thresholds by index when the sigmoid value of a multi-label class is considered as predicted
        
    Returns
    -------
    2D array
        The predictions encoded as defined by encoding
    """
    
    import numpy as np
    
    if classification_type == "multi-label":
        y_pred = np.zeros(probs.shape)
        
        prediction_probability_thresholds_by_index = dict(sorted(prediction_probability_thresholds_by_index.items()))
        unique_prediction_probability_thresholds = set(prediction_probability_thresholds_by_index.values())
        
        if len(unique_prediction_probability_thresholds) <= 1:
            prediction_probability_threshold = next(iter(unique_prediction_probability_thresholds), 0.5)
            y_pred[np.where(probs >= prediction_probability_threshold)] = 1
        else:
            for i in range(probs.shape[1]):
                prediction_probability_threshold = prediction_probability_thresholds_by_index.get(i, 0.5)
                mask = np.where(probs[i] >= prediction_probability_threshold)
                y_pred[i][mask] = 1
        
        if encoding == "one-hot":
            return y_pred

        indices = one_hot_to_indices(y_pred, classification_type=classification_type)

        if encoding == "index":
            return indices
        else: # encoding == "label":
            return indices_to_labels(indices, index2label)
        
    else:        
        y_pred = np.argmax(probs, axis=-1)
        if encoding == "index":
            return y_pred
        
        if encoding == "one-hot":
            return np.eye(num_classes)[y_pred]
        else: # encoding == "label":
            return indices_to_labels(y_pred, index2label)
