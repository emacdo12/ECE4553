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
    binary_labels = np.array([i == 1 for i in labels]).astype(int)
    return binary_labels,labels, dataset
    
def standardize_data(dataset):
    scaler = preprocessing.StandardScaler().fit(dataset)
    xscaled = scaler.transform(dataset)
    return xscaled, scaler

def QDA_Classify(dataset, labels, crossvalidation_dictionary,binary_labels):
    start_time = time.time()
    nfeatures = dataset.shape[1]
    nsamples  = dataset.shape[0]
    amount = crossvalidation_dictionary["amount"]
    percent = crossvalidation_dictionary["percent"]
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
        te_labels = binary_labels[te_ids]
        # train classifier
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(tr_data, tr_labels)
        predictions = qda.predict(te_data)
        for i in range(predictions.shape[0]):
            if predictions[i] != 1:
                predictions[i] = 0
        # get accuracy
        TP,TN,FP,FN,acc = get_results(predictions,te_labels)
        tp.append(TP)
        tn.append(TN)
        fp.append(FP)
        fn.append(FN)
        accuracy.append(acc)

        # accuracy += sum(predictions == te_labels)/te_labels.shape[0] / amount
        # average the accuracy
        results = np.average(accuracy)
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
    ULDA = np.loadtxt("ULDA_scaler.csv", delimiter=",")
    PCA = np.loadtxt("PCA_data.csv", delimiter=",")
    binary_labels,labels, dataset = load_dataset(data_folder)
    features = [39, 58, 8, 50, 22, 37, 9, 45, 40, 38, 49, 20, 25, 52, 10, 15, 59, 13, 7, 0, 46, 1, 14, 23, 21, 16, 68, 17, 61, 2] # SFFS
    # features = [13,10,12,52,22,66,68,18,23,69,36,17,50,11,44,1,20,38,39,46,27,0,28,25,16,40,41,37,21,7] # Fisher
    # features = [50,41,61,69,60,24,0,13,62,49,59,18,40,53,64,6,33,52,65,44,20,38,14,5,68,58,22,29,12,23] # MRMR
    # features = [50,40,39,38,57,5,58,0,12,52,55,4,37,10,59,6,51,8,18,14,1,35,13,23,15,34,9,20,16,33] # Info gain
    # features = [50,41,61,69,60,24,0,13,62,49,39, 58, 8, 22, 37, 9, 45, 40,20,38,57,5,12,52,55,4,10,59]
    # features = [50,41,40,13,10,12,39,58,22,37,9,45,40,38,49,20,25,61,69,60,24,0,62,53,57,5,4,8,59]
    # features = [39, 58, 8, 50, 22, 37, 9, 45, 40, 38, 49, 20, 25, 52, 10, 15, 59, 13, 7, 0, 46, 1, 14, 23, 21, 16, 68, 17, 61, 2, 54, 3, 34, 18, 5, 41, 44, 11, 36, 69, 4, 65, 56, 26, 19, 6, 62, 24, 28, 33]
    ULDA_vector =  np.matmul(dataset , ULDA)
    #features = list(range(1,14))
    pcadataset = PCA[:,0:5]
    #features = [0, 1, 2, 3, 4, 5]
    
    scaled_data, scaler = standardize_data(dataset)
    newdataset = scaled_data[:,features] 
    crossvalidation_dictionary = {
        "amount": 5,
        "percent": 0.75
    }
    results, runtime = QDA_Classify(newdataset, labels, crossvalidation_dictionary,binary_labels)
    print('Features: ' + str(features))
    print('Accuracy: ' + str(results))
    print('Time (in minutes): ' + str(runtime))
    
if __name__ == "__main__":
    main()