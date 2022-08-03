""" Main use method, class for use in the ReachClassifier software. Written by Emily Nguyen, Brett Nelson, Nicholas Chin,
    Lawrence Berkeley national labs / U C Berkeley."""
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import joblib
import pdb
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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


# Public, change sooon

def adjust_class_imbalance(X, y):
    """
    Adjusts for class imbalance.
        Object to over-sample the minority class(es) by picking samples at random with replacement.
        The dataset is transformed, first by oversampling the minority class, then undersampling the majority class.
    Returns: new samples
    References: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
    """
    oversampler = SMOTE(random_state=42)
    # undersampler = RandomUnderSampler(random_state=42)
    steps = [('o', oversampler)]  # , ('u', undersampler)]
    pipeline = Pipeline(steps=steps)
    X_res, y_res = pipeline.fit_resample(X, y)
    return X_res, y_res


def filter_y(df, y_name):
    """
    Separates X and y in df
    df: Df
    y_name: str col name
    Returns X and y separted
    """
    new_df = df.copy()
    for col in new_df.columns:
        if (not "PC" in col) and (col != y_name):
            new_df = new_df.drop([col], axis=1)
    X = new_df.drop([y_name], axis=1)
    y = new_df[y_name]  # .to_frame()
    return new_df, X, y


def plotModelsvAcc(data, y_name, models):
    # define data holding lists
    all_chance = []
    all_trainAcc = []
    all_valAcc = []
    all_ROC = []
    all_ROCAUC = []
    names = []

    # for each # of PCs 1, 2...n to use
    for modelInfo in models:
        model1, model2, name = modelInfo
        classifier(data, 'Trial Type').main_PCvROC(model1, model2)

        # do classification
        chance_score, train_score, val_score, roc, roc_auc = classifier(data, y_name).main_PCvROC(model1, model2)
        all_chance.append(chance_score)
        all_trainAcc.append(train_score)
        all_valAcc.append(val_score)
        all_ROC.append(roc)
        all_ROCAUC.append(roc_auc)
        names.append(name)

    # plot
    fig = plt.figure()
    #plt.ylim([0.4, 1.0]) # y axis range fixed to be 0.40 to 1.00
    plt.title(f"Models V Accuracy: {y_name}")
    x_axis = np.arange(len(models))
    width = 0.2
    plt.bar(x_axis+width, all_chance, width, label="chance")
    plt.bar(x_axis+width*2, all_trainAcc, width, label="train")
    plt.bar(x_axis+width*3,  all_valAcc, width, label="val")
    plt.bar(x_axis+width*4,  all_ROCAUC, width, label="roc auc")
    plt.xticks(x_axis+width, names)
    plt.legend()
    plt.savefig(f"ModelsvAcc: {y_name}")

    fig = plt.figure()
    plt.title(f"Models V ROC: {y_name}")
    for i in range(0, len(models)):
        fpr, tpr = all_ROC[i]
        plt.plot(fpr, tpr, label=names[i])
    plt.legend()
    plt.savefig(f"ModelsvROC: {y_name}")

class classifier:
    """ Methods to build , train, and interrogate a specific input set of classification algorithms. """

    def __init__(self, data, y_name):
        self.data = data
        self.y_name = y_name
    ### Adjust Class Imbalances ###

    def plot_scatter(X, y):
        """
        Return Scatter Plot in 2D
        X: Df
        y: Series
        """
        # summarize class distribution
        counter = y.value_counts()
        print(counter)

        # scatter plot of examples by class label
        plt.figure()
        plt.scatter(X["PC1"].values, X["PC2"].values, c=y.values)
        plt.show()

    def plot_scatter_3D(X, y):
        """
        Return Scatter Plot
        X: Df
        y: Series
        """
        a = X["PC1"].values
        b = X["PC2"].values
        c = X["PC3"].values

        # Creating figure
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")

        # Creating plot
        ax.scatter3D(a, b, c, c=y.values)
        plt.title("PCA 3D scatter plot")

        # show plot
        plt.show()

    ### Visualize Class Balances ###

    def plot_class_balance(y, convert_dict):
        counts = y.replace(convert_dict).value_counts()
        x_labels = list(convert_dict.values())
        plt.figure()
        plt.bar(x_labels, counts.values)
        plt.title("Label Class Distribution");

    ### Split ###

    def split_train_test(X, Y, split_percent=0.33, random_state=10):
        """ Function to split dataset into training and testing splits. """
        X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=split_percent, random_state=random_state)
        return X_temp, X_test, Y_temp, Y_test

    def split_train_validate_test(self, X, Y, split_percent=0.5, random_state=10):
        """ Function to split test data into testing and validation sets for hyperparameter tuning of models."""
        X_valid, X_test, y_valid, y_test = train_test_split(X, Y, test_size=split_percent, random_state=random_state)
        return X_valid, X_test, y_valid, y_test

    def find_chance(y):
        """ Shuffles y using the random.shuffle function, returns shuffled y vector."""
        y = y.copy()
        y = shuffle(y, random_state=42)
        return y.reset_index(drop=True)

    def tune_hyperparams(X_train, Y_train, random_state = 10):
        # Tuning hyperparameters for RandomForestClassifier
        # may take a bit to run
        # reference https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/

        # creat model
        model = RandomForestClassifier(random_state=random_state)

        # define hyperparameter to tune
        max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]  # make smaller to reduce runtime if necessary

        # define cross validation grid search
        grid = dict(max_depth=max_depth)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',
                                   error_score=0)
        grid_result = grid_search.fit(X_train, np.ravel(Y_train))

        # summarize results
        print("Best Accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        params = grid_result.cv_results_['params']

        # plot
        plt.plot([p['max_depth'] for p in params], means, marker='o', color='orange')
        plt.title("Random Forest Classifier: Max Depth of Trees vs Accuracy")
        plt.xlabel("n_estimators (Max Depth of Trees)")
        plt.ylabel("CV Accuracy")

        # define best n_estimators
        best_max_depth = grid_result.best_params_['max_depth']
        return best_max_depth

    def best_k_features(X_train, Y_train, n_estimators):
        # Visualize Accuracy vs Number of Features
        # may take a bit to run
        accuracies = []
        n = X_train.shape[1] + 1  # num cols
        for k in np.arange(1, n):
            selector = SelectKBest(score_func=f_classif,
                                   k=k)  # f_classif: ANOVA F-value between label/feature for classification tasks.
            z = selector.fit_transform(X_train, np.ravel(Y_train))  # temp new X

            # create, fit, and score model on k features
            model = RandomForestClassifier(n_estimators=n_estimators,
                                           random_state=10)  # use best n_estimators from previous part
            model.fit(z, np.ravel(Y_train))
            scores = cross_val_score(model, z, np.ravel(Y_train), cv=3)  # 3-fold cv to reduce time to run
            accuracies.append(np.mean(scores))

        # print best k
        a_dictionary = dict(zip(np.arange(1, n), accuracies))
        max_key = max(a_dictionary, key=a_dictionary.get)
        print(f"max accuracy {max(accuracies)} with k={max_key}.")

        # plot
        plt.figure()
        plt.plot(np.arange(1, n), accuracies, marker='v', color='green')
        plt.title("Random Forest Classifier: Number of Features vs Accuracy")
        plt.xlabel("k (# of most important features used)")
        plt.ylabel("CV Accuracy");
        plt.xticks(np.arange(1, n));
        plt.show()

        # define best k
        k = max_key
        return k

    def vis_feat_importance(X_train, Y_train):
        # Visualize best features
        selector = SelectKBest(score_func=f_classif,
                               k=3)  # f_classif: ANOVA F-value between label/feature for classification tasks.
        z = selector.fit_transform(X_train, np.ravel(Y_train))  # temp new X and k
        filter_imp = selector.get_support()  # mask for selected features
        features = np.array(X_train.columns)  # all feature names

        # create series for plotting
        forest_importances = pd.Series(selector.scores_, index=features)

        # sort in descending order
        forest_importances = forest_importances.sort_values(ascending=False)

        # print top 3 features
        print(f"3 Most Important Features: {features[filter_imp]}")

        # plot
        fig, ax = plt.subplots()
        forest_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances")
        ax.set_ylabel("ANOVA F-value (f_classif)")
        ax.set_xlabel("Features");

        return selector

        ### Scoring ###

    def score_model( X_train, Y_train, model, isPrint=True):
        # model.fit(X_train, np.ravel(Y_train))
        score = np.mean(cross_val_score(model, X_train, np.ravel(Y_train), scoring='accuracy', cv=5))
        if isPrint:
            print(f"Avg cv accuracy: {np.round(score, 3)}")
        # metrics.f1_score(np.ravel(Y_val), model.predict(X_val))
        return score

    def plot_ROC( X_test, Y_test, model):
        # ROC CURVE
        # calculate the fpr and tpr for all thresholds of the classification
        probs = model.predict_proba(X_test)
        preds = probs[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(np.ravel(Y_test), preds)
        roc_auc = metrics.auc(fpr, tpr)

        # plot ROC
        plt.figure()
        plt.title('Receiver Operating Characteristic Curve: Final RandomForestClassifier')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.008, 1])
        plt.ylim([0, 1.01])  # changed to better view curve at 1.0
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def main(self, convert_dict, doSMOTE=True, sn_k=True):
        """
        doSMOTE: bool, True to run, false otherwise
        """
        # get X, y
        n_data, n_X, n_y = filter_y(self.data, self.y_name)

        # adjust for class balances
        if doSMOTE:
            # smote
            sn_X, sn_y = adjust_class_imbalance(n_X, n_y)
        else:
            sn_X, sn_y = [n_X, n_y]

        # plot class balances
        classifier.plot_class_balance(n_y, convert_dict)
        classifier.plot_scatter(sn_X, sn_y)
        classifier.plot_scatter_3D(sn_X, sn_y)

        # split
        # sn_X_train, sn_X_val, sn_Y_train, sn_Y_val, sn_X_test, sn_Y_test = classifier.split(sn_X, sn_y)
        sn_X_train, sn_X_val, sn_Y_train, sn_Y_val, = classifier.split(sn_X, sn_y)

        # find chance
        model = RandomForestClassifier(random_state=0)
        shuffled_sn_y_train = classifier.find_chance(sn_Y_train)
        classifier.score_model(sn_X_train, shuffled_sn_y_train, model)

        # Compare to No tuning
        model = RandomForestClassifier(n_estimators=150, random_state=0)
        classifier.score_model(sn_X_train, sn_Y_train, model)

        # tune hyperparameters
        sn_n_estimators = 150  # Fix hyperparam at 150 trees to save compute time
        sn_max_depth = None  # classifier.tune_hyperparams(sn_X_train, sn_Y_train) # depth of trees #classifier.tune_hyperparams(sn_X_train, sn_Y_train, model)

        # visualize feature importances
        # sn_selector = classifier.vis_feat_importance(sn_X_train, sn_Y_train)

        # select best k features
        if sn_k:
            sn_k = classifier.best_k_features(sn_X_train, sn_Y_train, sn_n_estimators)
        else:
            sn_k = sn_X_train.shape[1]  # all feat

        # After tuning: create, fit, and score model on training data
        # selector = SelectKBest(score_func=f_classif, k=sn_k)  # f_classif: ANOVA F-value between label/feature for classification tasks.
        # z = selector.fit_transform(sn_X_val, np.ravel(sn_Y_val))  # new X
        # model = RandomForestClassifier(n_estimators=sn_n_estimators, max_depth=sn_max_depth, random_state=10)
        # model.fit(sn_X_train, sn_Y_train)
        # classifier.score_model(z, sn_Y_val, model)
        # model.fit(z, np.ravel(sn_Y_train))
        # score = np.mean(cross_val_score(model, z, np.ravel(sn_Y_train), cv=5))
        # print(f"Training avg cv training accuracy: After Tuning {np.round(score, 3)}")

        # evaluate using best hyperparams
        # temp_X = selector.transform(sn_X_val)
        # classifier.plot_ROC(temp_X, sn_Y_val, model)

        # tuning score
        model = RandomForestClassifier(n_estimators=sn_n_estimators, max_depth=sn_max_depth, random_state=10)
        # Fit on all training data
        model.fit(sn_X_train, sn_Y_train)
        classifier.score_model(sn_X_val, sn_Y_val, model)

        # ROC
        classifier.plot_ROC(sn_X_val, sn_Y_val, model)

    def main_PCvROC(self, model, model2):
        # get X, y
        n_data, n_X, n_y = filter_y(self.data, self.y_name)

        # adjust for class balances
        # smote
        sn_X, sn_y = adjust_class_imbalance(n_X, n_y)

        # split
        # sn_X_train, sn_X_val, sn_Y_train, sn_Y_val, sn_X_test, sn_Y_test = classifier.split(sn_X, sn_y)
        sn_X_train, sn_X_val, sn_Y_train, sn_Y_val, = classifier.split(sn_X, sn_y)

        # standardize
        scaler = StandardScaler().fit(sn_X_train)
        sn_X_train = scaler.transform(sn_X_train)
        sn_X_val = scaler.transform(sn_X_val)

        # normalize
        norm = Normalizer().fit(sn_X_train)
        sn_X_train = norm.transform(sn_X_train)
        sn_X_val = norm.transform(sn_X_val)

        # find chance
        shuffled_sn_y_train = classifier.find_chance(sn_Y_train)
        chance_score = classifier.score_model(sn_X_train, shuffled_sn_y_train, model, isPrint=False)

        # Compare to No tuning
        model = model2  # create new model
        train_score = classifier.score_model(sn_X_train, sn_Y_train, model, isPrint=False)  # fit and score on training

        # validation set
        model.fit(sn_X_train, sn_Y_train)
        val_score = classifier.score_model(sn_X_val, sn_Y_val, model, isPrint=False)

        # roc auc
        X_test = sn_X_val
        Y_test = sn_Y_val
        probs = model.predict_proba(X_test)
        preds = probs[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(np.ravel(Y_test), preds)
        roc_auc = metrics.auc(fpr, tpr)

        print(f"train acc: {train_score}, val acc: {val_score}, roc_auc: {roc_auc}")
        return chance_score, train_score, val_score, [fpr, tpr], roc_auc

    def plotPCAROC(self, data, y_name, maximum):
        """Produces plots of # of given PCs to a model vs the model's training, validation, and test accuracies.
        """
        # define data holding lists
        all_chance = []
        all_trainAcc = []
        all_valAcc = []
        all_ROC = []
        all_ROCAUC = []
        min = 3
        max = maximum
        NumPCsArray = range(min, max)

        # for each # of PCs 1, 2...n to use
        for n in NumPCsArray:
            # get y
            y = data[y_name]

            # select first n PCs
            df = data.iloc[:, :n]
            print(f"Num PCs: {n}")

            # append labels
            df = pd.concat([df, y], axis=1)

            # make models
            model = RandomForestClassifier(random_state=0)  # create new model
            model2 = RandomForestClassifier(n_estimators=150, random_state=0)  # create new model

            # do classification
            chance_score, train_score, val_score, roc, roc_auc = classifier(df, y_name).main_PCvROC(model, model2)
            all_chance.append(chance_score)
            all_trainAcc.append(train_score)
            all_valAcc.append(val_score)
            all_ROC.append(roc)
            all_ROCAUC.append(roc_auc)

        # plot
        plt.figure()
        plt.ylim([0.4, 1.0])  # y-axis range fixed to be 0.40 to 1.00
        plt.title(f"PCs V Accuracy: {y_name}")
        plt.plot(NumPCsArray, all_chance, label="chance")
        plt.plot(NumPCsArray, all_trainAcc, label="train")
        plt.plot(NumPCsArray, all_valAcc, label="val")
        plt.legend()
        plt.savefig(f"PCvAcc: {y_name}")

        plt.figure()
        plt.title(f"PCs V ROC: {y_name}")
        for i in range(0, len(all_ROC), 10):
            fpr, tpr = all_ROC[i]
            plt.plot(fpr, tpr, label=i + min)
        # plot last one
        fpr, tpr = all_ROC[-1]
        plt.plot(fpr, tpr, label=maximum)
        plt.legend()
        plt.savefig(f"PCvROC: {y_name}")

        plt.figure()
        plt.ylim([0.7, 1.0])  # y-axis range to be 0.70-1.00
        plt.title(f"PCs V ROC AUC: {y_name}")
        plt.plot(NumPCsArray, all_ROCAUC)
        plt.savefig(f"PCvROCAUC: {y_name}")
