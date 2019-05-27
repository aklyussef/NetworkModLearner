import os
import sys
#Load librariestry:
import pandas
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
import logging

class ModelData:

    def __init__(self,filepath,threshold,l):
        self.logger = l
        self.dataset = pandas.read_csv(filepath)
        self.dataset = self.dataset._get_numeric_data()
        self.labels = list(self.dataset.columns)
        #TODO: Accept splits from ModelData constructor
        self.threshold = threshold
        #Splits variable for k-fold cross validation
        self.splits = 5

        #create classification labeled dataframe
        self.l_dataset  = self.get_labeled_df(self.threshold,self.labels[-1])
        self.c_labels   = list(self.l_dataset.columns)
        return

    def get_labeled_df(self,cutoff,label):
        rdf = self.dataset.copy()
        rdf['class'] = rdf.apply(lambda row: 1 if row[label] >= cutoff else 0, axis=1)
        rdf.drop(label,axis=1,inplace=True)
        return rdf

    def show_data_features(self):
        #Show original dataset head
        print(self.dataset.head())
        self.logger.info(self.dataset.head)
        dataset = self.l_dataset
        #shape
        print(dataset.shape)
        self.logger.info(dataset.shape)
        #head
        print(dataset.head(20))
        self.logger.info(dataset.head(20))
        #statistical info on dataset
        print(dataset.describe())
        self.logger.info(dataset.describe())
        #class distribution
        print(dataset.groupby('class').size())
        self.logger.info(dataset.groupby('class').size())
        return

    def format_metric(self,metric):
        " formats tuple in form of (avg_metric,std) into percentages and 2 decimal points"
        avg,std = metric
        avg *= 100
        std *= 100
        return "({:.2f},{:.2f})".format(avg,std)

    def fit_and_evaluate(self):
        dataset = self.l_dataset
        array = dataset.values
        X = array[:,1:len(self.c_labels)-1]
        Y = array[:,len(self.c_labels)-1]
        validation_size = 0.20
        seed = 7
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                        random_state=seed)
        # scoring = 'accuracy'
        scoring = ['accuracy','recall','f1']

        self.models = []
        self.models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        self.models.append(('LDA', LinearDiscriminantAnalysis()))
        self.models.append(('KNN', KNeighborsClassifier()))
        self.models.append(('CART', DecisionTreeClassifier()))
        self.models.append(('NB', GaussianNB()))
        self.models.append(('SVM', SVC(gamma='auto')))
        # evaluate each model in turn
        self.results = []
        self.names = []
        self.model_eval_dict = {}
        for name, model in self.models:
            self.model_eval_dict[name] = []
            kfold = model_selection.KFold(n_splits=self.splits, random_state=seed)
            cv_results = model_selection.cross_validate(model,X_train,Y_train,cv=kfold,scoring=scoring)
            # cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
            self.results.append(cv_results['test_accuracy'])
            self.names.append(name)
            for metric in scoring:
                metric_name = '_'.join(['test',metric])
                self.model_eval_dict[name].append( (cv_results[metric_name].mean(),cv_results[metric_name].std()) )
                # msg = "%s: %s %f (%f)" % (name, metric_name,cv_results[metric_name].mean(), cv_results[metric_name].std())
                # print(msg)

        format_separator = ','
        info_str = '{:.2f} {}'.format(self.threshold,[x for x in dataset.groupby('class').size()])
        print(info_str)
        self.logger.info(info_str)
        header = 'Algorithm' + format_separator + format_separator.join(scoring) + format_separator + 'threshold'
        print(header)
        self.logger.info(header)
        for name in self.model_eval_dict.keys():
            metrics = name + format_separator + format_separator.join(self.format_metric(x) for x in self.model_eval_dict[name])
            print(metrics)
            self.logger.info(metrics)

        # for name,model in self.models:
        #     model.fit(X_train, Y_train)
        #     predictions = model.predict(X_validation)
        #     print(name+' accuracy score: ' + str(accuracy_score(Y_validation, predictions)))
        #     print(confusion_matrix(Y_validation, predictions))
        #     print(classification_report(Y_validation, predictions))
        return

    def show_comparison(self):
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(self.results)
        ax.set_xticklabels(self.names)
        plt.show()
        return

def init_logger():
    LOG_FORMAT = '%(message)s'
    logging.basicConfig(filename='ModelData.log',level=logging.DEBUG,filemode='w',format=LOG_FORMAT)
    l  = logging.getLogger()
    return l

# TODO: Consider using dataframe format to alleviate need for parsing data
def compare_algos_thresholds(filename):
    models = ['LR','LDA','KNN','CART','NB','SVM']
    metrics = ['accuracy','recall','f1']
    threshold_metrics = {}
    f = open(filename,'r')
    threshold = 0
    for line in f.readlines():
        if 'Algorithm' in line:
            continue
        if '[' in line:
            threshold = line.split(' ')[0]
            threshold_metrics[threshold] = [[],[],[]]
            continue
        if ',' in line:
            line = line.replace('(','')
            line = line.replace(')','')
            parts = line.split(',')
            accuracy = parts[1]
            threshold_metrics[threshold][0].append(accuracy)
            recall  = parts[3]
            threshold_metrics[threshold][1].append(recall)
            f1  = parts[5]
            threshold_metrics[threshold][2].append(f1)
    fig = plt.figure()
    fig.suptitle('Algorithm Performance Plot')
    ax = fig.add_subplot(111)
    return

def main():
    # if len(sys.argv) != 3:
    #     print('USAGE {} PATH_TO_DATASET CUTOFF'.format(sys.argv[0]))
    #     exit(1)
    filepath = sys.argv[1]
    # cutoff = float(sys.argv[2])
    l = init_logger()
    cutoffs = np.arange(0.1,1,0.1)
    for cutoff in cutoffs:
        l.debug('Reading dataset: ',filepath )
        print('Reading dataset: ',filepath )
        l.debug('Using cutoff: ',cutoff)
        print('Using cutoff: ',cutoff)
        md = ModelData(filepath,cutoff,l)
        # md.show_data_features()
        md.fit_and_evaluate()
    # md.show_comparison()

if __name__ == '__main__':
    # main()
    compare_algos_thresholds('ModelData.log')