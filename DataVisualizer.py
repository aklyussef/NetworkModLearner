import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

class DataVisualizer:

    def __init__(self,filepath):
        self.data_file = filepath
        self.df = self.build_dataframe()
        self.labels = list(self.df.columns)
        return

    def build_dataframe(self):
        return pd.read_csv(self.data_file)

    def build_plots(self):
        figure_ctr = 0
        # Subtract two since we don't want the first and last labels
        n_of_plots = len(self.labels)-2
        out_label = self.labels[-1]
        fig, axs  = plt.subplots(1,n_of_plots,figsize=(5,10))
        for label in self.labels[1:-1]:
            self.df.plot(kind='scatter',x=label,y=out_label,ax=axs[figure_ctr])
            axs[figure_ctr].set_title(label + 'vs. ' + out_label)
            figure_ctr += 1
        plt.show()

    def plot_var_distributions(self):
        self.df.hist()
        plt.show()
        return

    def plot_var_box(self):
        # TODO: Figure out how to make layout (AxB) depending on diff df
        self.df.plot(kind='box',subplots=True,sharex=False,sharey=False)
        plt.show()
        return

    # TODO: consider using _get_numeric_data().columns for x& y tick labels
    def plot_corelation_matrix(self):
        df = self.df.iloc[:,1:]
        correlations = df.corr()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('feature correlations')
        cax = ax.matshow(correlations,vmin=-1,vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0,len(self.labels),1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(self.labels[1:])
        ax.set_yticklabels(self.labels[1:])
        plt.show()
        return
   
    def plot_scatterplot_matrix(self):
       scatter_matrix(self.df)
       plt.show()
       return

def main():
    dv = DataVisualizer('../data/output/network_summary.csv')
    print(dv.labels)
    print(len(dv.labels))
    dv.build_plots()
    dv.plot_corelation_matrix()
    dv.plot_scatterplot_matrix()

if __name__ == '__main__':
    main()
