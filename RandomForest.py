import numpy as np
import time
import os
from glob import glob
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

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
    
def RandomForest(dataset, labels):


    return results



def main():
    data_folder = 'data'
    labels, dataset = load_dataset(data_folder)
    results = RandomForest(dataset, labels)
    print(results)
if __name__ == "__main__":
    main()