import numpy as np
import time
import os
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
from skfeature.function.similarity_based import fisher_score
from glob import glob
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import pandas as pd

def fisher(data_set,labels):
    fisher_scores = fisher_score.fisher_score(data_set,labels)
    # Plot Results
    feat_importance = pd.Series(fisher_scores, labels.columns[0:len(labels.columns)-1])
    feat_importance.plot(kind='barh', color = 'teal')
    plt.show
    return fisher_scores

def CC(data_set,labels):
    importances = mutual_info_classif(data_set,labels)
    # Plot Results
    feat_importance = pd.Series(importances, labels.columns[0:len(labels.columns)-1])
    feat_importance.plot(kind='barh', color = 'teal')
    plt.show
    return importances



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

def main():
    data_folder = 'data'
    labels, dataset = load_dataset(data_folder)
    scaled_data = standardize_data(dataset)
    scores = CC(dataset,labels)

if __name__ == "__main__":
    main()