import tensorflow as tf 
from tensorflow import keras
import numpy as np
import time
import os
from sklearn import preprocessing 
import pandas as pd 
from glob import glob

def standardize_data(dataset):
    scaler = preprocessing.StandardScaler().fit(dataset)
    xscaled = scaler.transform(dataset)
    return xscaled, scaler

def load_dataset(data_folder):
    f = [y for x in os.walk(data_folder) for y in glob(os.path.join(x[0], '*.csv'))]
    dataset = []
    labels = []
    for fi in f:
        data = pd.read_csv(fi, delimiter=",")
        data = data.fillna(0)
        labels.append(data.iloc[:,-1].to_numpy())
        dataset.append(data.iloc[:,:-1].to_numpy())
    labels = np.concatenate(labels)
    dataset = np.concatenate(dataset)
    # check for inf
    inf_rows = [id for id, i in enumerate(dataset) if (i == np.inf).any()]
    dataset = np.delete(dataset, inf_rows, axis=0)
    labels = np.delete(labels, inf_rows, axis=0)
    # check for only one value
    one_value_columns = [id for id, i in enumerate(dataset.T) if (np.unique(i).shape[0]==1) ]
    dataset = np.delete(dataset, one_value_columns, axis=1)
    # check for infinities in dataset, throw out those rows
    labels = np.array([i == "BENIGN" for i in labels]).astype(int)
    return labels, dataset

def main():
    data_folder = 'data'
    labels, dataset = load_dataset(data_folder)
    scaled_data, scaler = standardize_data(dataset)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(18),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(2)
    ])
    model.compile(optimizer = "adam",loss="mean_squared_error",metrics="mean_squared_error")

    features = [51,14,11,13,53,10,12,1,2,3,8,9,69,46,47,41,39,40]
    select_data = dataset[:,features]
    nsamples = select_data.shape[0]
    np.random.seed(seed=5)
    id = np.arange(select_data.shape[0])
    np.random.shuffle(id)
    percent = 0.75
    tr_ids = id[:int(percent * nsamples)]
    te_ids = id[int(percent * nsamples):]
    tr_data = select_data[tr_ids,:]
    te_data = select_data[te_ids, :]
    tr_labels = labels[tr_ids]
    te_labels = labels[te_ids]

    history = model.fit(tr_data,tr_labels)
    predictions = model.evaluate(te_data,te_labels)
    print(predictions)
if __name__ == "__main__":
    main()




