
def get_compute_device():
    """Provides the device for the computation
    
    Returns
    -------
    str
        The GPU device with number (cuda:0) or cpu
    """
    
    #import torch
    #return "cuda:0" if torch.cuda.is_available() else "cpu"
        
    import tensorflow as tf
    return "cuda:0" if tf.config.list_physical_devices("GPU") else "cpu"


def gpu_empty_cache():
    """Cleans the GPU cache which seems to fill up after a while
    
    """
    
    import tensorflow as tf
    
    #if torch.cuda.is_available():
    if tf.config.list_physical_devices("GPU"):
        import torch
        torch.cuda.empty_cache()
        

def compute_predictions(params, data, predict_func):
    """Computes the actual predictions. Allows for recovery in case of a crash...

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    data: dataframe
        The data
    predict_func: callable
        The function that computes the prediction
    """
    
    import os
    from datetime import datetime
    
    from fhnw.nlp.utils.storage import save_dataframe
    from fhnw.nlp.utils.storage import load_dataframe
    
    verbose = params.get("verbose", False)
    batch_size = params.get("batch_size", 8)
    X_column_name = params.get("X_column_name", "text")
    y_column_name = params.get("y_column_name", "label")
    y_column_name_prediction = params.get("y_column_name_prediction", "prediction")
    store_every_n_elements = params.get("store_every_n_elements", 32768)
    store_path = params.get("store_path", "data/predictions.parq")
    last_stored_batch = params.get("last_stored_batch", -1)
    empty_gpu_cache = params.get("empty_gpu_cache", False)
    
    predictions = []
    
    # load stored data for recovery
    if last_stored_batch >= 0 or last_stored_batch == -1 and os.path.exists(store_path):
        predictions_loaded = load_dataframe(store_path)
        predictions = [row.to_dict() for index, row in predictions_loaded.iterrows()]
        
        if last_stored_batch < 0:
            last_stored_batch = len(predictions) // batch_size
            
        if verbose:
            print(datetime.now().time(), "Loaded batch:", last_stored_batch, " predictions: ", len(predictions))
         
    # do the predictions
    for g, df in data.groupby(np.arange(len(data)) // batch_size):
        if g >= last_stored_batch:
            # prevent OOM on GPU
            if empty_gpu_cache:
                gpu_empty_cache()
                
            predictions_batch = predict_func(params, df[X_column_name].to_list())
            
            # store the predictions together with the data
            i = 0
            for index, row in df.iterrows():
                # e.g. back translation might provide more than one translation per prediction
                if isinstance(predictions_batch[i], list):
                    for prediction in predictions_batch[i]:
                        row_dict = row.to_dict()
                        row_dict[y_column_name_prediction] = prediction
                        predictions.append(row_dict)
                else:
                    row_dict = row.to_dict()
                    row_dict[y_column_name_prediction] = predictions_batch[i]
                    predictions.append(row_dict)

                i += 1

                
            if (g + 1) % (store_every_n_elements // batch_size) == 0:
                if verbose:
                    print(datetime.now().time(), "Save batch:", str(g+1), ", processed elements:", str((g+1)*batch_size), ", total predictions:", len(predictions))

                save_dataframe(pd.DataFrame(predictions), store_path)

    if verbose:
        print(datetime.now().time(), "Prediction done. Batches:", str(data.shape[0] // batch_size), ", processed elements:", str(data.shape[0]), ", total predictions:", len(predictions))
    
    pred_data = pd.DataFrame(predictions)
    save_dataframe(pred_data, store_path)
    
    return pred_data
    

def dataframe_to_dataset(params, data):
    """Converts a dataframe into a Huggingface dataset

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    data: dataframe
        The data
        
    Returns
    -------
    dataset
        The Huggingface dataset
    """
    
    import pandas as pd
    from fhnw.nlp.utils.params import compute_binarized_labels
    from fhnw.nlp.utils.params import create_label_binarizer_and_set
    
    from datasets import ClassLabel
    from datasets import Dataset
    
    X_column_name = params.get("X_column_name", "text")
    y_column_name = params.get("y_column_name", "label")
    computed_objects_column_name = params.get("computed_objects_column_name", "computed_objects")
    
    data = data.drop(labels=data.columns.difference([X_column_name, y_column_name]), axis=1)
    
    label_binarizer = params.setdefault(computed_objects_column_name, {}).get("label_binarizer", None)
    if label_binarizer is None:
        create_label_binarizer_and_set(params, data)
        label_binarizer = params[computed_objects_column_name]["label_binarizer"]
    num_classes = len(label_binarizer.classes_)
    class_names = list(label_binarizer.classes_)
    
    data[y_column_name] = compute_binarized_labels(params, data)
    
    data.index.name = "idx"
    
    dataset = Dataset.from_pandas(data)
    
    dataset.features["label"] = ClassLabel(num_classes=num_classes, names=class_names)
    
    return dataset


def dataframe_to_datasets(params, data, data_test=None):
    """Converts a dataframe into a Huggingface dataset dictionary with a train, val and test split

    Parameters
    ----------
    params: dict
        The dictionary containing the parameters
    data: dataframe
        The data
    data_test: dataframe
        The user defined test split (use None to automatically generate a test split)  
        
    Returns
    -------
    dataset
        The Huggingface dataset dictionary
    """
    
    import pandas as pd
    from fhnw.nlp.utils.params import compute_binarized_labels
    from fhnw.nlp.utils.params import create_label_binarizer_and_set
    from fhnw.nlp.utils.params import get_train_test_split
    
    from datasets import ClassLabel
    from datasets import Dataset
    from datasets import DatasetDict
    
    X_column_name = params.get("X_column_name", "text")
    y_column_name = params.get("y_column_name", "label")
    computed_objects_column_name = params.get("computed_objects_column_name", "computed_objects")
    
    data = data.drop(labels=data.columns.difference([X_column_name, y_column_name]), axis=1)

    # ensure label_binarizer builds on all data (function dataframe_to_dataset will use the existing)
    label_binarizer = params.setdefault(computed_objects_column_name, {}).get("label_binarizer", None)
    if label_binarizer is None:
        create_label_binarizer_and_set(params, data)
        label_binarizer = params[computed_objects_column_name]["label_binarizer"]
    
    # do the split
    if data_test is None:
        data, data_test = get_train_test_split(params, data)
    data_train, data_val = get_train_test_split(params, data)

    dataset_train = dataframe_to_dataset(params, data_train)
    dataset_test = dataframe_to_dataset(params, data_test)
    dataset_val = dataframe_to_dataset(params, data_val)
    
    transformers_datasets = DatasetDict({"train":dataset_train, "validation":dataset_val, "test":dataset_test})
    
    params.setdefault(computed_objects_column_name, {})["transformers_datasets"] = transformers_datasets
    
    return transformers_datasets
