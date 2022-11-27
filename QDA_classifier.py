import numpy as np
import time
import os
from glob import glob
from sklearn import preprocessing
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
    for i in range(labels.shape[0]):
        if "Bot" in labels[i]:
            labels[i] = 2
        elif "DDoS" in labels[i]:
            labels[i] = 3
        elif "GoldenEye" in labels[i]:
            labels[i] = 4
        elif "Hulk" in labels[i]:
            labels[i] = 5
        elif "Slowhttptest" in labels[i]:
            labels[i] = 6
        elif "slowloris" in labels[i]:
            labels[i] = 7
        elif "FTP-Patator" in labels[i]:
            labels[i] = 8
        elif "Heartbleed" in labels[i]:
            labels[i] = 9
        elif "Infiltration" in labels[i]:
            labels[i] = 10
        elif "PortScan" in labels[i]:
            labels[i] = 11
        elif "SSH-Patator" in labels[i]:
            labels[i] = 12
        elif "Brute" in labels[i]:
            labels[i] = 13
        elif "Sql" in labels[i]:
            labels[i] = 14
        elif "XSS" in labels[i]:
            labels[i] = 15
        else:
            labels[i] = 1
    labels = labels.astype(int)   
    # comment two lines below to remove binary case
    #print('Binary case')
    #labels = np.array([i == "BENIGN" for i in labels]).astype(int)
    return labels, dataset
    
def standardize_data(dataset):
    scaler = preprocessing.StandardScaler().fit(dataset)
    xscaled = scaler.transform(dataset)
    return xscaled, scaler

def QDA_Classify(dataset, labels, crossvalidation_dictionary):
    start_time = time.time()
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
    end_time = (time.time() - start_time)/60
    return results, end_time



def main():
    data_folder = 'data'
    labels, dataset = load_dataset(data_folder)
    features = [39, 58, 8, 50, 22, 37, 9, 45, 40, 38, 49, 20, 25, 52, 10, 15, 59, 13, 7, 0, 46, 1, 14, 23, 21, 16, 68, 17, 61, 2] 
    scaled_data, scaler = standardize_data(dataset)
    newdataset = scaled_data[:,features] 
    crossvalidation_dictionary = {
        "amount": 5,
        "percent": 0.75
    }
    results, runtime = QDA_Classify(newdataset, labels, crossvalidation_dictionary)
    print('Features: ' + str(features))
    print('Accuracy: ' + str(results))
    print('Time (in minutes): ' + str(runtime))
    
if __name__ == "__main__":
    main()