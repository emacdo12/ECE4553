import numpy as np
import time
import os
from glob import glob
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
    #labels = np.array([i == "BENIGN" for i in labels]).astype(int)
    return labels, dataset
    


def sequential_forward_selection(dataset, labels, crossvalidation_dictionary):
    nfeatures = dataset.shape[1]
    nsamples  = dataset.shape[0]
    amount = crossvalidation_dictionary["amount"]
    percent = crossvalidation_dictionary["percent"]
    results    = np.zeros((nfeatures, nfeatures))
    results[:] = np.nan
    best_order = []
    perm_featureset = np.array([])
    # split dataset <amount> times
    #for iteration in range(nfeatures):
    for iteration in range(10):
        start_time = time.time()
        # your feature set has <iteration> permanent features

        for fi in range(nfeatures):
            if fi in best_order:
                continue
            # add fi to our feature set to test
            fi_featureset = np.hstack((perm_featureset, dataset[:,fi].reshape(-1,1))) if perm_featureset.size else np.expand_dims(dataset[:,fi], axis=1)
            accuracy = 0
            for k in range(amount):
                np.random.seed(seed=k)
                id = np.arange(nsamples)
                np.random.shuffle(id)
                tr_ids = id[:int(percent * nsamples)]
                te_ids = id[int(percent * nsamples):]
                tr_data = fi_featureset[tr_ids,:]
                te_data = fi_featureset[te_ids, :]
                tr_labels = labels[tr_ids]
                te_labels = labels[te_ids]
                # train classifier
                lda = LinearDiscriminantAnalysis()
                lda.fit(tr_data, tr_labels)
                predictions = lda.predict(te_data)
                # get accuracy
                accuracy += sum(predictions == te_labels)/te_labels.shape[0] / amount
            # average the accuracy
            results[iteration, fi] = accuracy
        # choose the best feature for this iteration
        best_feature = np.nanargmax(results[iteration,:])
        best_order.append(best_feature)
        perm_featureset = np.hstack((perm_featureset, dataset[:,best_feature].reshape(-1,1))) if perm_featureset.size else np.expand_dims(dataset[:,best_feature], axis=1)
        end_time = (time.time() - start_time)/60
        print('Iteration #: ' + str(iteration) + '\nBest Feature: ' + str(best_feature) + '\nTime (in mins): ' + str(end_time))

    return results, best_order



def main():
    data_folder = 'data'
    labels, dataset = load_dataset(data_folder)
    
    crossvalidation_dictionary = {
        "amount": 5,
        "percent": 0.75
    }
    results, order = sequential_forward_selection(dataset, labels, crossvalidation_dictionary)
    print(order)
if __name__ == "__main__":
    main()