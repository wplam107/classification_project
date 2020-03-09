import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import scipy.stats as stats

class ModelDF:
    def __init__(self, df, cat_features=[], target=''):
        self.df = df.copy()
        self.cat_features = cat_features
        self.target = target

    def cat_and_drop(self):
        print('---')
        try:
            for feat in self.cat_features:
                self.df[feat] = self.df[feat].astype('category')
                self.df = self.df.join(pd.get_dummies(self.df[feat], prefix='{}'.format(feat), drop_first=True))
                self.df.drop(columns=feat, inplace=True)
                print('Added dummies for and dropped "{}"'.format(feat))
            print('Now has {} columns'.format(self.df.shape[1]))
        except:
            print('No dummies added and no columns dropped')

    def _check_imbalance(self, col):
        '''
        Check class imbalance of target or feature
        Parameters
        ----------
        col = str, column to check class imbalance
        '''
        return self.df[col].value_counts(normalize=True), self.df[col].value_counts()

    def info(self):
        print('---')
        print('Shape: {}'.format(self.df.shape))
        for i in range(len(self.df.dtypes.unique())):
            print('There is/are {} {} feature(s)'.format(self.df.dtypes.value_counts()[i],
                                                    self.df.dtypes.value_counts().index[i]))
        a, b = self._check_imbalance(col=self.target)
        print('---')
        print('Target Variable Class Ratios:\n{}'.format(a))
        print('Target Variable Counts:\n{}'.format(b))

    def new_cat(self, new_feat, old_feat, bin_point=0, equality='e'):
        '''
        Create new categorical feature from old feature
        Parameters
        ----------
        new_feat : str, name of new feature to be created
        old_feat : str, reference feature
        bin_point : int or float, point of binning, default 0
        equality : str, 'ge' is >=, 'g' is >, 'e' (default) is ==, 'le' is <=, 'l' is <
        '''
        if equality == 'e':
            self.df[new_feat] = np.where(self.df[old_feat] == bin_point, 1, 0)
        elif equality == 'ge':
            self.df[new_feat] = np.where(self.df[old_feat] >= bin_point, 1, 0)
        elif equality == 'g':
            self.df[new_feat] = np.where(self.df[old_feat] > bin_point, 1, 0)
        elif equality == 'le':
            self.df[new_feat] = np.where(self.df[old_feat] <= bin_point, 1, 0)
        else:
            self.df[new_feat] = np.where(self.df[old_feat] < bin_point, 1, 0)

    def preprocess(self, major, minor, test_size=0.2, random_state=0, samp_type=None, scaler=None):
        '''
        Get train-test split and resample data
        Parameters
        ----------
        test_size : float between 0 and 1, default = 0.2
        random_state : int, default = 0
        samp_type : str or None
            'up' 'down' 'smote' 'tomek'
        scaler : str or None
            'standard' 'minmax'
        '''
        self._tts(test_size=test_size, random_state=random_state)
        self._resample(samp_type=samp_type, random_state=random_state, major=major, minor=minor)
        self._scaler(scaler=scaler)


    def _getXy(self, X=None, y=None):
        '''
        Get X (features) and y (target)
        Parameters
        ----------
        X : List of strings
            features, default uses all columns
        y : Target variable
            default uses self.target
        '''
        if X == None:
            self.X = self.df.drop(self.target, axis=1)
        else:
            self.X = self.df[X]
        if y == None:
            self.y = self.df[self.target]
        else:
            self.y = self.df[y]
        print('X and y acquired')

    def _tts(self, test_size=0.2, random_state=0):
        '''
        Train test split on DataFrame
        Parameters
        ----------
        test_size : float between 0 and 1, default = 0.2
        random_state : int, default = 0
        '''
        self._getXy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                                self.X, self.y, test_size=test_size, random_state=random_state)
        print('Data has been split into train and test sets')

    def _resample(self, major, minor, random_state=0, samp_type=''):
        '''
        Resample for binary class imbalance
        Parameters
        ----------
        samp_type : str
             'up' 'down' 'smote' 'tomek'
        '''
        df = pd.concat([self.X_train, self.y_train], axis=1)
        major, minor = df[df[self.target] == major], df[df[self.target] == minor]

        if samp_type == 'up':
            print('Data upsampled')
            self._simple_resample(minor, major, random_state)
        elif samp_type == 'down':
            print('Data downsampled')
            self._simple_resample(major, minor, random_state)
        elif samp_type == 'smote':
            print('SMOTE performed')
            self._smote_data(random_state)
        elif samp_type == 'tomek':
            print('Tomek Links performed')
            self._tomek_data()
        else:
            print('No Resampling performed')

    def _simple_resample(self, change, goal, random_state):
        resampled = resample(change, replace=True, n_samples=len(goal), random_state=random_state)
        joined = pd.concat([goal, resampled])
        self.X_train, self.y_train = joined.drop(self.target, axis=1), joined[self.target]

    def _smote_data(self, random_state):
        sm = SMOTE(random_state=random_state)
        self.X_train, self.y_train = sm.fit_sample(self.X_train, self.y_train)

    def _tomek_data(self):
        tl = TomekLinks()
        self.X_train, self.y_train = tl.fit_sample(self.X_train, self.y_train)

    def _scaler(self, scaler=None):
        if scaler == 'standard':
            scale = StandardScaler()
            scale.fit_transform(self.X_train)
            scale.transform(self.X_test)
        elif scaler == 'minmax':
            scale = MinMaxScaler()
            scale.fit_transform(self.X_train)
            scale.transform(self.X_test)
        else:
            print('No Scaling performed')
    
    def _push_out(self):
        '''
        Removal of features with a score of 0 importance
        '''
        push = self.xgb_model.fit(self.X_train, self.y_train)
        worthless = [ self.X_train.columns[idx] for idx, val in enumerate(push.feature_importances_) if val == 0 ]
        self.X_train = self.X_train.drop(columns=worthless)
        self.X_test = self.X_test.drop(columns=worthless)
        num_pushed = len(worthless)
        print('---')
        print('Number of Features Removed: {}'.format(num_pushed))

    def get_xgb(self, gs=False, params=None, push_out=False):
        '''
        Instantiate and fit XGBoost model object with or without GridSearch to train set
        Default model as .xgb_model and GridSearch model as .gs_xgb

        Parameters
        ----------
        gs : bool
            True = XGBoost with GridSearch CV
            False (default) = default XGBoost
        params : dictionary
            parameters to run through GridSearch CV
        push_out : bool
            True = remove features with no importance and 
            Warning alters X_train and X_test features
        '''
        if gs == True:
            xgb_model = xgb.XGBClassifier()
            self.gs_xgb = GridSearchCV(
                            estimator=xgb_model,
                            param_grid=params,
                            scoring='f1',
                            n_jobs=-1,
                            verbose=1,
                            cv=5)
            self.gs_xgb.fit(self.X_train, self.y_train)
            self.xgb_model = self.gs_xgb.best_estimator_
            preds = self.xgb_model.predict(self.X_test)

            test_f1 = f1_score(self.y_test, preds)
            test_acc = accuracy_score(self.y_test, preds)

            print("Accuracy: %f" % (test_acc))
            print("F1: %f" % (test_f1))
            print('Best Parameters:\n{}'.format(self.gs_xgb.best_params_))

            if push_out == True:
                self._push_out()
                self.gs_xgb = GridSearchCV(
                            estimator=xgb_model,
                            param_grid=params,
                            scoring='f1',
                            n_jobs=-1,
                            verbose=1,
                            cv=5)
                self.gs_xgb.fit(self.X_train, self.y_train)
                self.xgb_model = self.gs_xgb.best_estimator_
                self.xgb_model.fit(self.X_train, self.y_train)

                preds = self.xgb_model.predict(self.X_test)

                test_f1 = f1_score(self.y_test, preds)
                test_acc = accuracy_score(self.y_test, preds)

                print("Accuracy After Push Out: %f" % (test_acc))
                print("F1 After Push Out: %f" % (test_f1))
                print('Best Parameters After Push Out:\n{}'.format(self.gs_xgb.best_params_))

        else:
            self.xgb_model = xgb.XGBClassifier().fit(self.X_train, self.y_train)
            self.xgb_model.fit(self.X_train, self.y_train)
        
            if push_out == True:
                self._push_out()
                self.xgb_model.fit(self.X_train, self.y_train)

            preds = self.xgb_model.predict(self.X_test)

            test_f1 = f1_score(self.y_test, preds)
            test_acc = accuracy_score(self.y_test, preds)

            print("Accuracy: %f" % (test_acc))
            print("F1: %f" % (test_f1))
    
    def _best_feats(self):
        features = [ (self.X_train.columns[idx], round(val, 4))
                    for idx, val in enumerate(self.xgb_model.feature_importances_)
                    if val != 0 ]
        best = sorted(features, key=lambda x:x[1], reverse=True)[:10]
        return best
    
    def plot_bf(self):
        '''
        Bar Plot of top 10 features in XGBoost
        '''
        best = pd.DataFrame(self._best_feats(), columns=['Features', 'Importance'])
        f, ax = plt.subplots(figsize = (25,5))
        sns.barplot(x='Features', y='Importance', data=best)
        plt.show()

    def get_rf(self, gs=False, params=None):
        '''
        Instantiate and fit RandomForest model object with or without GridSearch to train set
        Default model as .rf_model and GridSearch model as .gs_rf
        Parameters
        ----------
        gs : bool
            True = RandomForest with GridSearch CV
            False (default) = default RandomForest
        params : dictionary
            parameters to run through GridSearch CV
        '''
        if gs == True:
            self.rf_model = RandomForestClassifier()
            self.gs_rf = GridSearchCV(
                            estimator=self.rf_model,
                            param_grid=params,
                            scoring='f1',
                            n_jobs=-1,
                            verbose=1,
                            cv=5)
            self.gs_rf.fit(self.X_train, self.y_train)

            preds = self.gs_rf.best_estimator_.predict(self.X_test)

            test_f1 = f1_score(self.y_test, preds)
            test_acc = accuracy_score(self.y_test, preds)

            print("Accuracy: %f" % (test_acc))
            print("F1: %f" % (test_f1))
            print('Best Parameters:\n{}'.format(self.gs_rf.best_params_))

        else:
            self.rf_model = RandomForestClassifier().fit(self.X_train, self.y_train)
            self.rf_model.fit(self.X_train, self.y_train)

            preds = self.rf_model.predict(self.X_test)

            test_f1 = f1_score(self.y_test, preds)
            test_acc = accuracy_score(self.y_test, preds)

            print("Accuracy: %f" % (test_acc))
            print("F1: %f" % (test_f1))

# Functions for EDA and Feature Engineering

def multi_plot(df, plot='hist', target=''):
    '''
    Plotting continuous features for EDA
    Parameters
    ----------
    df : DataFrame
    type : str
        'hist' as histogram or 'lmplot' as lmplot
    target : str
        target variable
    '''
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            if plot == 'hist':
                df.hist(col)
                plt.show()
            elif plot == 'lmplot':
                sns.lmplot(x=col, y=target, data=df, logistic=True)
                plt.show()

def colin_plt(df, target='', context='poster', figsize=(20,10), ft_scale=0.7):
    sns.set(rc = {'figure.figsize':figsize})
    sns.set_context('poster', font_scale=ft_scale)
    sns.heatmap(df.drop(target, axis=1).corr(), cmap='Reds', annot=True)
    plt.show()

def chi_sq(df, feature='', target='', bin_point=0):
    '''
    Chi-Squared test for single feature, uses alpha = 0.05 and ddof = 1
    Parameters
    ----------
    feature : str
        feature column to inspect as str
    target : str
        target variable
    bin_point : int or float, default = 0
        the equal or less than point where to bin as int or float
    '''
    def _bin_bin(df):
        el_bin_t = len(df.loc[(df[feature] <= bin_point) & (df[target] == True)])
        el_bin_f = len(df.loc[(df[feature] <= bin_point) & (df[target] == False)])
        g_bin_t = len(df.loc[(df[feature] > bin_point) & (df[target] == True)])
        g_bin_f = len(df.loc[(df[feature] > bin_point) & (df[target] == False)])
        return el_bin_t, el_bin_f, g_bin_t, g_bin_f
    
    el_t, el_f, g_t, g_f = _bin_bin(df)

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