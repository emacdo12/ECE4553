import numpy as np
import time
import os
from glob import glob
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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
    # comment two lines below to remove binary case
    print('Binary case')
    labels = np.array([i == "BENIGN" for i in labels]).astype(int)
    return labels, dataset
    


def QDA_Classify(dataset, labels, crossvalidation_dictionary):
    nfeatures = dataset.shape[1]
    nsamples  = dataset.shape[0]
    amount = crossvalidation_dictionary["amount"]
    percent = crossvalidation_dictionary["percent"]
    accuracy = 0
    for k in range(amount):
        np.random.seed(seed=k)
        id = np.arange(nsamples)
        np.random.shuffle(id)
        tr_ids = id[:int(percent * nsamples)]
        te_ids = id[int(percent * nsamples):]
        tr_data = dataset[tr_ids,:]
        te_data = dataset[te_ids, :]
        tr_labels = labels[tr_ids]
        te_labels = labels[te_ids]
        # train classifier
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(tr_data, tr_labels)
        predictions = qda.predict(te_data)
        # get accuracy
        accuracy += sum(predictions == te_labels)/te_labels.shape[0] / amount
        # average the accuracy
        results = accuracy
    return results



def main():
    data_folder = 'data'
    labels, dataset = load_dataset(data_folder)
    features = [50,41,61,69,60,24,0,13,62,49]
    newdataset = dataset[:,features]
    crossvalidation_dictionary = {
        "amount": 5,
        "percent": 0.75
    }
    results = QDA_Classify(newdataset, labels, crossvalidation_dictionary)
    print('Features: ' + str(features))
    print('Accuracy: ' + str(results))
if __name__ == "__main__":
    main()