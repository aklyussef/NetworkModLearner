import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

class DataVisualizer:

    def __init__(self,filepath):
        self.data_file = filepath
        self.full_dataframe = self.build_dataframe()
        self.labels = list(self.full_dataframe.columns)

        self.numeric_data = self.full_dataframe._get_numeric_data()
        self.numeric_columns = self.numeric_data.columns
        return

    def dataframe_summary(self):
        for column in self.numeric_columns:
            self.get_stats(self.numeric_data,column)
        return

    def get_labeled_df(self,cutoff,label):
        rdf = self.numeric_data.copy()
        rdf['label'] = rdf.apply(lambda row: 1 if row[label] >= cutoff else 0, axis=1)
        rdf.drop(label,axis=1,inplace=True)
        return rdf

    def get_stats(self,df,label):
        stars = '*' * 10
        print(stars + '\t{} stats\t'.format(label) + stars)
        print('Mean', df[label].mean())
        print('Median',df[label].median())
        print('Max',df[label].max())
        print('Min',df[label].min())
        print('std_dev',df[label].std())
        # print(stars + '\t' + stars + '\t' + stars)
        return

    def build_dataframe(self):
        return pd.read_csv(self.data_file)

    def build_plots(self):
        self.build_plot('scatter')
        self.build_plot('box')
        return

    def build_plot(self,kind=None):
        figure_ctr = 0
        # df = self.get_labeled_df(0.3,self.numeric_data[-1])
        df = self.numeric_data
        labels = self.numeric_columns
        out_label = labels[-1]

        # compute how many rows and columns we need for the variables
        import math
        n_col  = 3
        n_rows = math.ceil((len(labels)-1)/n_col)

        col_ctr = 0
        row_ctr = 0
        fig, axs  = plt.subplots(n_rows,n_col,figsize=(5,10))
        for label in labels:
            if kind == 'scatter':
                self.numeric_data.plot(kind=kind, x=label, y=out_label, ax=axs[row_ctr][col_ctr])
            elif kind == 'box':
                self.numeric_data.boxplot(column=label, ax=axs[row_ctr][col_ctr])
            # axs[row_ctr][col_ctr].set_title(label + 'vs. ' + out_label)
            row_ctr = (row_ctr+1) % n_rows
            if(row_ctr == 0):
                col_ctr = (col_ctr+1) % n_col
        plt.show()
        return

    def plot_var_distributions(self):
        self.numeric_data.hist()
        plt.show()
        return

    def plot_corelation_matrix(self):
        df = self.numeric_data
        correlations = df.corr()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('feature correlations')
        cax = ax.matshow(correlations,vmin=-1,vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0,len(self.numeric_columns),1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        s_labels = self.get_summarized_labels(self.numeric_columns)
        ax.set_xticklabels(s_labels)
        ax.set_yticklabels(s_labels)
        plt.show()
        return

    def get_summarized_labels(self,labels):
        summarized_labels = []
        for string in labels:
            summarized_labels.append(string[:6])
        return summarized_labels

    def plot_scatterplot_matrix(self):
       scatter_matrix(self.full_dataframe)
       plt.show()
       return

def main():
    # dv = DataVisualizer('../data/output/network_summary.csv')
    dv = DataVisualizer('../data/output/network_summary.csv')
    dv.dataframe_summary()

    dv.build_plots()
    dv.plot_var_distributions()
    dv.plot_corelation_matrix()
    dv.plot_scatterplot_matrix()

if __name__ == '__main__':
    main()
