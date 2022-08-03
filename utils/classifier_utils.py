""" Utility functions for use in the ReachClassifier software. Written by Emily Nguyen, Brett Nelson, Nicholas Chin,
    Lawrence Berkeley national labs / U C Berkeley."""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import joblib
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing
import sklearn.metrics as metrics
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import AdaBoostClassifier


# Public functions

def load_saved_csv_dataframes(path):
    """ Utility function to load .csv format stored dataframes using Pandas.
        :param path: path to file
        :type path: str
        :return dataframe: dataframe containing training, test, or pre-processed reaching data.
        :type dataframe: pandas dataframe """
    dataframe = pd.read_csv(path)
    return dataframe


def set_seed(ntype=None):
    """Utility function to set a random or int-set seed.
        :param ntype: variable seed
        :type ntype: int
    """
    if ntype:
        random.seed(ntype)
    else:  # default value
        random.seed(47)
    return


def preprocess_labels(labels):
    """ Function to preprocess labels (alpha currently)
        :param labels: set of classification labels
        :type labels: pandas dataframe
        :return new_labels: new, preprocessed labels
        :type new_labels: pandas dataframe"""
    # Remove unmatched labels (manual)
    new_labels = labels.loc[(labels['Rat'] != "RM13") | (labels["Session"] != "S3") | (labels['Date'] != 20)]
    # Remove non-class columns
    new_labels.reset_index(inplace=True, drop=True)
    return new_labels


def pre(discrete_df, standardize=False, drop_cols=True):
    """ Function to pre-process input data by removing possible columns of nuisance, standardize the data,
        return a dataframe.
        :param discrete_df: pandas DataFrame containing data
        :type discrete_df: pandas dataframe
        :param standardize: flag to use the sci-kit learn StandardScalar function to standardize our data using a
                            gaussian distribution
        :type standardize: bool
        :param drop_cols: flag option to drop pesky columns
        :type drop_cols: bool
        :return discrete_df: new, preprocessed dataframe
        :type discrete_df: pandas dataframe
    """
    if drop_cols:
        for col in discrete_df.columns:
            if (col == "Unnamed: 0") or ("_rob_p50" in col) or ("_sensor_p50" in col):
                discrete_df = discrete_df.drop(col, axis=1)
    if standardize:
        discrete_df2 = StandardScaler().fit_transform(discrete_df)
        discrete_df = pd.DataFrame(discrete_df2, columns=discrete_df.columns)
    return discrete_df


def doPCA(df, n, labels, concat=True):
    """
    Performs PCA on df.
    Returns dataframe with PCA columns and pca object.
    """
    pca = PCA(n_components=n)
    components = pca.fit_transform(df)
    PCA_col_names = ["PC" + str(x) for x in range(1, pca.n_components_ + 1)]
    pca_df = pd.DataFrame(components, columns=PCA_col_names)
    if concat:
        pca_df = pd.concat([pca_df, labels], axis=1)
    return pca_df, pca


def save_PC_components(model, filename):
    """ Function to save a .joblib extension file that contains pre-specified amount of principle components.
    :param model: a model from sk-learn PC component function containing vectorized representations of PC components.
    :type model: numpy array of length N_PC's X Trial
    :param filename: save path from top of directory
    """
    joblib.dump(model, filename)
    return


def generalize_data_to_labels(df_extended_labels):
    """ This function creates multiple new datasets from an extended behavioral dataset, depending on type of trial.
    Datasets are returned."""
    notNull_data = df_extended_labels[df_extended_labels['Trial Type'] == 0]  # 1 if null, 0 if real trial
    Null_data = df_extended_labels[df_extended_labels['Trial Type'] == 1]
    # single v multiple datasets
    singleReach_data = notNull_data[notNull_data['Num Reaches'] == 0]  # 0 if <=1, 1 if > 1 reaches
    multipleReach_data = notNull_data[notNull_data['Num Reaches'] == 1]

    # need to excludes bin and combine lr
    rightHand_data = singleReach_data[singleReach_data["Which Hand"] == 5]  # 5 for right, 1 for left
    leftHand_data = singleReach_data[singleReach_data["Which Hand"] == 1]

    return notNull_data, Null_data, singleReach_data, multipleReach_data, rightHand_data, leftHand_data



