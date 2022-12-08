import numpy as np
import time
import os
from glob import glob
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
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

def standardize_data(dataset):
    scaler = preprocessing.StandardScaler().fit(dataset)
    xscaled = scaler.transform(dataset)
    return xscaled


def MLP_Classify(dataset, labels, crossvalidation_dictionary):
    start_time = time.time()
    nfeatures = dataset.shape[1]
    nsamples  = dataset.shape[0]
    amount = crossvalidation_dictionary["amount"]
    percent = crossvalidation_dictionary["percent"]
    accuracy = 0
    accuracy = []
    tp = []
    tn = []
    fp = []
    fn = []
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
        clf = MLPClassifier(solver='adam',verbose=True, random_state=1, tol=1e-4, n_iter_no_change=5, max_iter=70)
        clf.fit(tr_data, tr_labels)
        predictions = clf.predict(te_data)
        # get accuracy
        TP,TN,FP,FN,acc = get_results(predictions,te_labels)
        tp.append(TP)
        tn.append(TN)
        fp.append(FP)
        fn.append(FN)
        accuracy.append(acc)
        #accuracy += sum(predictions == te_labels)/te_labels.shape[0] / amount
        # average the accuracy
        results = accuracy
        print("Iteration Complete!")
    end_time = (time.time() - start_time)/60
    print('TP: ' + str(np.average(tp)))
    print('FP: ' + str(np.average(fp)))
    print('TN: ' + str(np.average(tn)))
    print('FN: ' + str(np.average(fn)))
    print('Accuracy: ' + str(np.average(accuracy)))
    print('STD: ' + str(np.std(accuracy)))
    std = np.std(accuracy)
    return results, end_time

def get_results(pred_labels,true_labels):
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range (true_labels.shape[0]):
        if true_labels[i] == pred_labels[i]:
            if pred_labels[i] == 1:
                a = a + 1
            else:
                d = d + 1
        else:
            if pred_labels[i] == 1:
                c = c + 1
            else:
                b = b + 1
    
    TP = d/(c+d)
    TN = a/(a+b)
    FP = b/(a+b)
    FN = c/(c+d)
    acc = (a+d)/(a+b+c+d)

    return TP,TN,FP,FN,acc

def main():
    data_folder = 'data'
    labels, dataset = load_dataset(data_folder)
    scaled_data = standardize_data(dataset)
    crossvalidation_dictionary = {
        "amount": 5,
        "percent": 0.75
    }
    results, runtime = MLP_Classify(scaled_data, labels, crossvalidation_dictionary)
    print('Accuracy: ' + str(results))
    print('Time (in minutes): ' + str(runtime))
if __name__ == "__main__":
    main()