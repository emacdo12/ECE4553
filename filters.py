import numpy as np
import time
import os
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from glob import glob
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import pandas as pd

def info_gain(data_set,labels):
    importances = mutual_info_classif(data_set,labels)
    return importances

def CC(data_set,labels):
    df = pd.DataFrame(data_set)
    cor = df.corr()
    plt.figure()
    sns.heatmap(cor)
    plt.show()
    return cor

def random_forest_importance(data_set,labels):
    df = pd.DataFrame(data_set)
    model = RandomForestClassifier(n_estimators=340)
    model.fit(data_set,labels)
    importances = model.feature_importances_

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

def chi_squared(dataset, labels):
    nfeatures = dataset.shape[1]
    for i in range(nfeatures):
        if dataset[1,i] < 0:
            dataset = np.delete(dataset, i, axis = 1)
    data_cat = dataset.astype(int)
    chi2_features = SelectKBest(chi2, k = 10)
    best_features = chi2_features.fit_transform(data_cat, labels)
    num_features = best_features.shape[1]
    return best_features, num_features

def main():
    data_folder = 'data'
    labels, dataset = load_dataset(data_folder)
    importances = info_gain(dataset,labels)
    cor = CC(dataset,labels)
    forest_importances = random_forest_importance(dataset,labels)

    # Write to csv file
    information_gain = np.asarray(importances)
    np_cor = np.asarray(cor)
    np_forest_importances = np.asarray(forest_importances)

    np.savetxt("Information_gain.csv",information_gain, delimiter=",")
    np.savetxt("Correlation.csv",np_cor, delimiter=",")
    np.savetxt("Forest_Importance.csv",np_forest_importances, delimiter=",")



if __name__ == "__main__":
    main()