import os
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

class ModelData:

    def __init__(self,filepath,threshold):
        self.dataset = pandas.read_csv(filepath)
        self.dataset = self.dataset._get_numeric_data()
        self.labels = list(self.dataset.columns)
        #TODO: Accept splits from ModelData constructor
        self.threshold = threshold
        #Splits variable for k-fold cross validation
        self.splits = 5
        return

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
        dataset = self.l_dataset
        #shape
        print(dataset.shape)
        #head
        print(dataset.head(20))
        #statistical info on dataset
        print(dataset.describe())
        #class distribution
        print(dataset.groupby('class').size())
        return

    def fit_and_evaluate(self):
        dataset = self.l_dataset
        array = dataset.values
        X = array[:,1:len(self.c_labels)-1]
        Y = array[:,len(self.c_labels)-1]
        validation_size = 0.20
        seed = 7
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                        random_state=seed)
        scoring = 'accuracy'

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
        for name, model in self.models:
            kfold = model_selection.KFold(n_splits=self.splits, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
            self.results.append(cv_results)
            self.names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

        for name,model in self.models:
            model.fit(X_train, Y_train)
            predictions = model.predict(X_validation)
            print(name+' accuracy score: ' + str(accuracy_score(Y_validation, predictions)))
            print(confusion_matrix(Y_validation, predictions))
            print(classification_report(Y_validation, predictions))
        return

    def show_comparison(self):
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(self.results)
        ax.set_xticklabels(self.names)
        plt.show()

def main():
    md = ModelData('../network_summary.csv',0.3)
    md.show_data_features()
    md.fit_and_evaluate()

if __name__ == '__main__':
    main()