import requests
import json
import pandas as pd
import os
import glob

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import sys
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn import neighbors, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

def check_exist_lang(lang,r, wiki_id):
    #checks if exists a wikipedia link of a given language of a given subject
    wiki = lang+"wiki"
    return wiki in r.json()['entities'][wiki_id]['sitelinks']

def find_files(source_L="en", target_L="pt"):
    #On this function we gonna find the missing articles of a target language given a source languange.
    #This could be a general function, but for this project we gonna reduce our search, looking only for english files that misses
    #portuguese files on wikipedia.
    itens_ID_exists=[]
    itens_ID_dont_exists=[]
    error_cont=0

    #for j_file in ["queryP4466.json", "queryP118.json"]:
    for j_file in ["queryP4466.json"]:

        with open(j_file) as f:
            wiki = json.load(f)

        wikidata_concepts=list(set([item['item'].split("/")[-1] for item in wiki if item['item'].split("/")[-1][0] == "Q"]))

        for wiki_id in wikidata_concepts:
            #vars for names of the wikipedia files in pt and eng
            eng_name=""
            pt_name=""
            #colect data that we need to identify if this subject already have pt information
            command = "api.php?action=wbgetentities&ids="+wiki_id+"&redirects=no&format=json"
            try:
                r = requests.get('https://www.wikidata.org/w/'+command)
            except Exception as e:
                r = ""
                print "error in url"
                error_cont+=1

            if r!="" and r.json()['success']:
                if (check_exist_lang(source_L, r, wiki_id)):
                    eng_name=r.json()['entities'][wiki_id]['sitelinks']['enwiki']['title'].replace(" ","_")
                    if (check_exist_lang(target_L, r, wiki_id)):
                        eng_name=r.json()['entities'][wiki_id]['sitelinks']['ptwiki']['title'].replace(" ","_")
                        itens_ID_exists.append([wiki_id, eng_name, pt_name])
                    else:
                        itens_ID_dont_exists.append([wiki_id, eng_name, pt_name])
                #print (r.json()['entities'])

    #save all those ids in a file for backup if necessary
    # with open("wiki_br.txt", "w") as f:
    #     f.write(itens_ID_exists)
    # with open("wiki_no_br.txt", "w") as f:
    #     f.write(itens_ID_dont_exists)

    print "total errors: "+str(error_cont)
    return itens_ID_exists, itens_ID_dont_exists

def search_index_views(iten, data,has_pt):
    eng_views=data[data['wiki_name'] == iten[1]]['views'].unique()[0]
    if has_pt:
        pt_views=data[data['wiki_name'] == iten[2]]['views'].unique()[0]
        return eng_views, pt_views
    return eng_views

def associate_page_views(itens_ID_exists, itens_ID_dont_exists):
    page_views_file = open("page_views.csv", "a")
    missing_rank_file = open("missing_rank.csv", "a")

    files = glob.glob('pageview-11-05-2015/*')
    #second column convert to datetimeindex
    dfs = [pd.read_csv(fp, sep=" ",index_col=[1], header=None) for fp in files]
    data = pd.concat(dfs).sort_index()
    data.columns = ["lang", "wiki_name", "views", "size"]

    for i in itens_ID_exists:
        eng_views, pt_views=search_index_views(i, data,1)
        line=str(i[0])+" "+str(eng_views)+" "+str(pt_views)
        page_views_file.append(line)
    for i in itens_ID_dont_exists:
        eng_views=search_index_views(i, data,0)
        line=str(i[0])+" "+str(eng_views)
        missing_rank_file.append(line)



def create_array_model():
    myfile = open("page_views.csv")
    mylines = myfile.read().split('\n')
    #y is the value that we want to predict, so y is the pt_page_views
    #X is the array of the features, we only have one feature, the en+page_views
    X=[]
    y=[]
    for line in mylines[1:len(mylines)]:
        element = line.split(";")
        #using try only to make sure that the program won't stop if a variable of the data set has some problem
        try:
            y.append(int(element[-1]))
            X.append([float(x) for x in element[1:-2]])
    X = np.array(X)
    y = np.array(y)
    return X,y

def rank_articles():
    X,y=create_array_model()
    kn=10

    kf = KFold(kn, shuffle=True)
    metrics = [0,""]

    Classifiers=["LR", "KNN", "RF", "NB","XGBoost"]

    #this loop will interate for each classifier using diferents kfolds
    for classifier in Classifiers:
        for train_index, test_index in kf.split(X):
            print "Classifier: ",classifier
            print "using kfold= ",kn
            print "\n"
            #this will chose the classifier, and use gridSearch to choose the best hyper parameters focussing on reach the best AUC score
            # Linear Regression
            if classifier == "LR":
              Cs = [ 10.0**c for c in xrange(-2, 3, 1) ]
              clf = GridSearchCV(estimator=LogisticRegression(), param_grid=dict(C=Cs), scoring="roc_auc", n_jobs=-1, cv=5, verbose=0)
            # K Neighbors
            elif classifier == "KNN":
              Ks = [ k for k in xrange(1, 15, 2) ]
              clf = GridSearchCV(estimator=neighbors.KNeighborsClassifier(), param_grid=dict(n_neighbors=Ks), scoring="roc_auc", n_jobs=-1, cv=5, verbose=0)
            #Nayve Bayes
            elif classifier == "NB":
              clf = GridSearchCV(estimator=GaussianNB(), param_grid=dict(),scoring="roc_auc", n_jobs=-1, cv=5, verbose=0)
            #random forest
            elif classifier == "RF":
              estimators = [ e for e in xrange(5, 25, 5) ]
              clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=dict(n_estimators=estimators), scoring="roc_auc", n_jobs=-1, cv=5, verbose=0)
            #XGBoost
            elif classifier == "XGBoost":
              clf = xgb.XGBClassifier()

            #this fit will train your classifier to your dataset
            clf.fit(X[train_index],y[train_index])
            #this will return the probability of each example be 0 or 1
            y_pred = clf.predict_proba(X[test_index])
            #this will generate the AUC score
            auc = roc_auc_score(y[test_index], y_pred[:,1])
            print "AUC: ", str(auc)
            print "###########################\n"

            #this will save the greater value, the estimator (model), the train and test set, to reproduce the best model with the real train set file
            if (classifier!="XGBoost"):
                if metrics[0]<auc:
                    metrics=[auc, clf.best_estimator_, X[train_index], y[train_index]]
            else:
                if metrics[0]<auc:
                    metrics=[auc, "xgboost",X[train_index], y[train_index]]

    #he must use the classifier with the best score
    if metrics[1]=="xgboost":
        clf_greater=xgb.XGBClassifier()
    else:
        clf_greater=metrics[1]
    clf_greater.fit(metrics[2], metrics[3])


    #start to classify the missing data file
    file_test = open("missing_rank.csv")

    print "generating test labels!!\n"
    mylines_test = file_test.read().split('\n')
    #tratando arquivo de teste
    X_test = []
    signal = 1
    for line in mylines_test[1:len(mylines_test)]:
        element = line.split(";")
        X_test.append([float(x) for x in element[1:-1]])
    X_test = np.array(X_test)

    probs = clf_greater.predict_proba(X_test)[:,1]

    for idx,prob in enumerate(probs):
        mylines_test[idx].append(prob)

    print mylines_test.sort(key=lambda x:x[-1])


#def match_editors():

if __name__ == '__main__':
    itens_ID_exists, itens_ID_dont_exists = find_files()
    associate_page_views(itens_ID_exists, itens_ID_dont_exists)
    rank_articles()
