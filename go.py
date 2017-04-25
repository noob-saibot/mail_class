from data_extraction import Extractor, Learning
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator, RegressorMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import ClassifierMixin
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels

# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

class CustomEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn
    df_train = pd.read_csv('x_train.csv', delimiter=';', header=None)
    df_test = pd.read_csv('x_test.csv', delimiter=';', header=None)
    df_sub = pd.read_csv('y_train.csv', delimiter=';', names=['Class'])
    df_rs = pd.concat([df_train, df_sub], axis=1)
    df_rs = pd.concat([df_rs, df_test], ignore_index=True)
    del df_test, df_train, df_sub

    df_train = df_rs.dropna()

    for i in df_train.columns:
        tips = df_train[i]
        print(i)
        Q = tips.quantile([.1, .5, .9]).values
        X2 = Q[0] - 1.5 * (Q[2] - Q[0])
        X1 = Q[2] + 1.5 * (Q[2] - Q[0])
        tips.ix[tips > X1] = tips.mean()
        tips.ix[tips < X2] = tips.mean()
        df_train[i] = tips

    lst_of_low_corr = [
        4, 9, 14, 39, 40,
        56, 65, 86, 100, 132, 145, 149, 151,
        159, 161, 166, 173, 194, 196, 197, 202,
        203, 207
           ]
    lst_of_not_too_low_corr = [3, 24, 25, 38, 50, 53, 80, 192]
    lst_hight_self_corr_1 = [102, 103, 104, 105, 106, 107]

    df_train = df_train.drop(lst_of_low_corr, axis=1)
    df_train = df_train.drop(lst_of_not_too_low_corr, axis=1)

    az = seaborn.heatmap(df_train.corr())
    plt.show()

    # add spec and drop list of self correlated features 1
    df_train['spec1'] = df_train[lst_hight_self_corr_1].sum(axis=1)/len(lst_hight_self_corr_1)
    df_train = df_train.drop(lst_hight_self_corr_1, axis=1)

    lst_hight_self_corr_2 = [140, 141, 142, 143, 144, 146, 147, 148]
    # add spec and drop list of self correlated features 2
    df_train['spec2'] = df_train[lst_hight_self_corr_2].sum(axis=1) / len(lst_hight_self_corr_2)
    df_train = df_train.drop(lst_hight_self_corr_2, axis=1)

    lst_hight_self_corr_3 = [180, 181, 182, 183, 184, 185, 186, 187, 188]
    # add spec and drop list of self correlated features 3
    df_train['spec3'] = df_train[lst_hight_self_corr_3].sum(axis=1) / len(lst_hight_self_corr_3)
    df_train = df_train.drop(lst_hight_self_corr_3, axis=1)


    for i in df_train.columns:
        if i != 'Class':
            #df_train[i] = (df_train[i]-df_train[i].mean(axis=0))/df_train[i].max(axis=0)
            df_train[i] = df_train[i] / df_train[i].max(axis=0)


    # this block lead to bad result ib beta
    # from sklearn.feature_selection import f_classif
    # m = f_classif(df_train.drop(['Class'],axis=1),
    #                                      df_train['Class'])
    # imp = list(zip(m[0],df_train.drop(['Class'], axis=1).columns))
    #
    #
    # az = seaborn.heatmap(df_train.corr())
    # for i in imp:
    #     print(i)
    #     if i[0]<2:
    #         df_train = df_train.drop([i[1]], axis=1)
    #
    # from sklearn.decomposition import PCA
    #
    # pca = PCA(n_components=20)
    # y_t = df_train['Class']
    # rs_sc = pca.fit_transform(df_train[[i for i in df_train.columns if i != 'Class']])
    # rs_sc_df = pd.DataFrame(rs_sc)
    # df_train = pd.concat([rs_sc_df, y_t], axis=1)
    #########


    az = seaborn.heatmap(df_train.corr())
    plt.show()


    Lst = ["from sklearn.ensemble import AdaBoostClassifier",
    "from sklearn.ensemble import BaggingClassifier",
    "from sklearn.ensemble import ExtraTreesClassifier",
    "from sklearn.ensemble import GradientBoostingClassifier",
    "from sklearn.ensemble import RandomForestClassifier",
    "from sklearn.ensemble import VotingClassifier",
    "from sklearn.gaussian_process import GaussianProcessClassifier",
    "from sklearn.linear_model import PassiveAggressiveClassifier",
    "from sklearn.linear_model import RidgeClassifier",
    "from sklearn.linear_model import RidgeClassifierCV",
    "from sklearn.linear_model import SGDClassifier",
    "from sklearn.multiclass import ClassifierMixin",
    "from sklearn.multiclass import OneVsOneClassifier",
    "from sklearn.multiclass import OneVsRestClassifier",
    "from sklearn.multiclass import OutputCodeClassifier",
    "from sklearn.neighbors import KNeighborsClassifier",
    "from sklearn.neighbors import RadiusNeighborsClassifier",
    "from sklearn.neural_network import MLPClassifier",
    "from sklearn.tree import DecisionTreeClassifier",
    "from sklearn.tree import ExtraTreeClassifier"]

    sp = []
    for i in Lst:
        i = i.split('import ')[1]
        try:
            rs = Learning(df_train, y_col='Class').trees({

                }, eval(i), scoring='accuracy')
            print(i, rs)
            if rs > 0.48:
                sp.append(i)
            fimp = eval(i+' ()').fit(df_train.drop(['Class'], axis=1), df_train.Class)
            print(list(zip(df_train.drop(['Class'], axis=1).columns, fimp.feature_importances_)))
        except Exception as e:
            print(e)

    for i in sp:
        locals()['My'+i] = eval(i)()

    print(Learning(df_train, y_col='Class').trees(
            m_params={'demo_param': [eval('My'+i) for i in sp]},
            models=CustomEnsembleClassifier,
            cross=True))