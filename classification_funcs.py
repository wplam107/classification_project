import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
import scipy.stats as stats

class ModelDF:
    def __init__(self, df, cat_features=[], target=''):
        self.df = df.copy()
        self.cat_features = cat_features
        self.target = target

    def _drop_columns(self, cols=[]):
        print('---')
        try:
            self.df.drop(columns=cols, inplace=True)
            print('List of columns dropped: {}'.format(cols))
        except:
            print('No columns dropped')

    def _cat_and_drop(self):
        print('---')
        for feat in self.cat_features:
            self.df[feat] = self.df[feat].astype('category')
            self.df = self.df.join(pd.get_dummies(self.df[feat], prefix='{}'.format(feat), drop_first=True))
            self.df.drop(columns=feat, inplace=True)
            print('Added dummies for and dropped "{}"'.format(feat))
        print('Now has {} columns'.format(self.df.shape[1]))

    def _info(self):
        print('---')
        print('Shape: {}'.format(self.df.shape))
        for i in range(len(self.df.dtypes.unique())):
            print('There are {} {} features'.format(self.df.dtypes.value_counts()[i],
                                                    self.df.dtypes.value_counts().index[i]))

    def _multi_plot(self, plot='hist'):
        '''
        Plotting continuous features for EDA
        Parameters
        ----------
        df : DataFrame
        type : str
            'hist' as histogram or 'lmplot' as lmplot
        target : str, optional
            target (dependent) variable column if lmplot
        '''

        for col in self.df.columns:
            if self.df[col].dtype == 'float64' or self.df[col].dtype == 'int64':
                if plot == 'hist':
                    self.df.hist(col)
                    plt.show()
                elif plot == 'lmplot':
                    sns.lmplot(x=col, y=self.target, data=self.df, logistic=True)
                    plt.show()

    def _chi_sq(self, feature='', bin_point=0):
        '''
        Chi-Squared test for single feature
        Parameters
        ----------
        feature : str
            feature column to inspect as str
        bin_point : int or float, default = 0
            the equal or less than point where to bin as int or float
        '''

        def _bin_bin(self):
            el_bin_t = len(self.df.loc[(self.df[feature] <= bin_point) & (self.df[self.target] == True)])
            el_bin_f = len(self.df.loc[(self.df[feature] <= bin_point) & (self.df[self.target] == False)])
            g_bin_t = len(self.df.loc[(self.df[feature] > bin_point) & (self.df[self.target] == True)])
            g_bin_f = len(self.df.loc[(self.df[feature] > bin_point) & (self.df[self.target] == False)])
            return el_bin_t, el_bin_f, g_bin_t, g_bin_f
        
        el_t, el_f, g_t, g_f = _bin_bin(self)

        tot_t = el_t + g_t
        tot_f = el_f + g_f
        tot_el = el_t + el_f
        tot_g = g_t + g_f

        ex_elt = tot_el *  tot_t/(tot_t+tot_f)
        ex_elf = tot_el *  tot_f/(tot_t+tot_f)
        ex_gt = tot_g * tot_t/(tot_t+tot_f)
        ex_gf = tot_g * tot_f/(tot_t+tot_f)
    
        chi, p = stats.chisquare([el_t, el_f, g_t, g_f], [ex_elt, ex_elf, ex_gt, ex_gf], ddof=1)
        if chi > 3.8415:
            print('Reject Null Hypothesis')
        else:
            print('Cannot Reject Null Hypothesis')
        print('Chi-Squared: {}'.format(chi))
        print('p-value: {}'.format(p))
