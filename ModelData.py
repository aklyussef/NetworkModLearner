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

    #TODO: Figure out how to change data into binary classification problem using threshold since this class only handles bianry/multiclass classification

    def __init__(self,filepath):
        self.dataset = pandas.read_csv(filepath)
        self.labels = list(self.dataset.columns)
        #TODO: Accept splits from ModelData constructor
        self.splits = 2
        return

    def show_data_features(self):
        #shape
        print(self.dataset.shape)
        #head
        print(self.dataset.head(20))
        #statistical info on dataset
        print(self.dataset.describe())
        #class distribution
        print(self.dataset.groupby(self.labels[-1]).size())
        return

    def fit_and_evaluate(self):
        array = self.dataset.values
        X = array[:,1:len(self.labels)-1]
        Y = array[:,len(self.labels)-1]
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


        def show_comparison(self):
            fig = plt.figure()
            fig.suptitle('Algorithm Comparison')
            ax = fig.add_subplot(111)
            plt.boxplot(self.results)
            ax.set_xticklabels(self.names)
            plt.show()


def main():
    md = ModelData('../data/output/network_summary.csv')
    md.show_data_features()
    md.fit_and_evaluate()

if __name__ == '__main__':
    main()

def to_be_integrated():
	# Make predictions on validation dataset
	knn = KNeighborsClassifier()
	knn.fit(X_train, Y_train)
	predictions = knn.predict(X_validation)
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))
