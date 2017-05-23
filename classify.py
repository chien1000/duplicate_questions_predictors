#! python3
# -*- coding:utf-8 -*-
import sys
import csv
from time import time
from preprocess import Qpair
import preprocess
import numpy as np 
from tqdm import tqdm

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib

def load_data(fname):
    data = []
    file = open(fname,'r',encoding='utf-8')
    for row in csv.DictReader(file):
        if 'is_duplicate' in row:
            qp = Qpair(row['question1'],row['question2'],row['is_duplicate'])
        else:
            qp = Qpair(row['question1'],row['question2'],-1)        
        data.append(qp)
    print("finish loading data")
    return(data)

def make_features(fname, train=True):
    data = load_data(fname)
    if train:
        all_text = []
        all_tokens = []
        for qp in data:
            all_text.append(qp.q1)
            all_text.append(qp.q2)
            t1 = [tup[0] for tup in qp.q1_tokenized]
            t2 = [tup[0] for tup in qp.q2_tokenized]
            all_tokens.append(t1)
            all_tokens.append(t2)

        preprocess.initialize(all_text, all_tokens)
    else:
        preprocess.initialize(None,None)

    with open("features_" + fname,'w') as foutput:
        foutput.write(",".join(Qpair.get_feature_names()))
        for i,qp in enumerate(tqdm(data, desc = 'make features and write to the file')):
            foutput.write("\n")            
            values = qp.get_features()
            if str(type(values)) == "<class 'scipy.sparse.csr.csr_matrix'>":
                values = values.todense().flat
            values = ["{:.2f}".format(v) for v in values]
            foutput.write(",".join(values))

def make_labels(fname):
    data = load_data(fname)    
    with open("label_" + fname,'w') as loutput:
        y = [qp.is_duplicate for qp in data]
        loutput.write("\n".join(map(str,y)))

def load_features(fname):
    fm = []
    with open("features_" + fname, 'r', encoding='utf-8') as ff:
        lines = ff.readlines()
        head = lines[0]
        # print(head)
        for line in tqdm(lines[1:], desc = 'loading features'):
            values = line.strip("\n").split(",")
            fm.append(values)
    fm = np.array(fm)
    return fm

def load_labels(fname):
    y = []
    with open("label_" + fname,'r') as lf:
        for line in lf.readlines():
            y.append(line.strip('\n'))
    y = np.array(y)

    return y

def train_evaluate(clf, X_train, X_test, y_train, y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    # import pdb; pdb.set_trace()
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score

def train(fname='train.csv', save_fname='model.pkl'):
    # data = load_data("features_train.csv")
    # N= len(data)
    # M= len(data[0].get_features())
    # X= np.zeros((N,M))
    # y= np.zeros(N,dtype=np.int)
    # for i,qp in enumerate(data):
    #     X[i,] = qp.get_features()
    #     y[i] = qp.is_duplicate
    X = load_features(fname)    
    y = load_labels(fname)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)

    joblib.dump(clf, save_fname)

def predict_test(fname="test.csv", clf_file='model.pkl'):
    clf = joblib.load(clf_file) 
    # data = {}
    # f = open(fname,'r',encoding='utf-8') 
    # for row in csv.DictReader(f):
    #     pair_id = int(row["test_id"])
    #     data[pair_id] = Qpair(row["question1"],row["question2"],-1)
    
    # N= len(data)
    # M= len(data[1].get_features())
    # X= np.zeros((N,M))
    # for i,qp in data.items():
    #     X[i,] = qp.get_features()
    X = load_features(fname)
    pred = clf.predict(X)

    from datetime import datetime
    now = datetime.now()
    output = "predict_" + now.strftime("%m_%d_%H_%M") + ".txt"
    with open(output,'w') as ofile:
        ofile.write("test_id,is_duplicate\n")
        for i, y in enumerate(pred):
            ofile.write("{},{}\n".format(i,y))
            
        
def evaluate(fname):
    # data = load_data("train.csv")
    # N= len(data)
    # M= len(data[0].get_features())
    # X= np.zeros((N,M))
    # y= np.zeros(N,dtype=np.int)
    # for i,qp in enumerate(data):
    #     X[i,] = qp.get_features()
    #     y[i] = qp.is_duplicate

    X = load_features(fname)    
    y = load_labels(fname)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=516)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # clf = LinearSVC()
        clf = RandomForestClassifier(n_estimators=100)
        # clf = LogisticRegression()
        train_evaluate(clf, X_train, X_test, y_train, y_test)
        
        clf = LinearSVC()
        train_evaluate(clf, X_train, X_test, y_train, y_test)
        
        clf = LogisticRegression()
        train_evaluate(clf, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    if len(sys.argv)>1:
        if sys.argv[1] == 'train':
            if len(sys.argv) >2:
                fname = sys.argv[2]
                train(fname)
            else:
                train()
                
        elif sys.argv[1] == 'predict':
            predict_test('test.csv')

        elif sys.argv[1] == 'evaluate':
            if len(sys.argv) >2:
                fname = sys.argv[2]
                evaluate(fname)
            else:
                evaluate()           

        elif sys.argv[1] == 'make_features':
            try:
                if sys.argv[2] == 'train':
                    make_features("train.csv")
                    make_labels("train.csv")
                elif sys.argv[2] == 'test':
                    make_features("test.csv", False)
                else:
                    fname = sys.argv[2]
                    make_features(fname)
                    make_labels(fname)
                    
            except Exception as e:
                print(str(e))