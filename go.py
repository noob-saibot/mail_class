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
from xgboost import XGBClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_selection import RFECV
from sklearn.cluster import KMeans

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
    import pickle
    # this mod part (raplcing outlier, nFeature scaling and mean normalization, demension reduction)
    df_train = pd.read_csv('x_train.csv', delimiter=';', header=None)
    df_test = pd.read_csv('x_test.csv', delimiter=';', header=None)
    print(df_test.shape)
    df_sub = pd.read_csv('y_train.csv', delimiter=';', names=['Class'])
    df_train = pd.concat([df_train, df_sub], axis=1)

    sp = []
    cor = df_train.corr()

    def corr_concat(df_train, sp, coef, df_test):
        for i in df_train.columns:
            sp_mod = []
            rs = cor[i][cor[i] > coef]
            if len(rs) != 1:
                for j in rs.index.values:
                    if j not in sp:
                        sp_mod.append(j)
            if sp_mod:
                print('Drop %s'%i)
                df_train['H%s'%i] = df_train[sp_mod].sum(axis=1) / len(sp_mod)
                df_test['H%s' % i] = df_test[sp_mod].sum(axis=1) / len(sp_mod)
                df_train = df_train.drop(sp_mod, axis=1)
            sp += list(rs.index.values)
        return df_train, df_test



    df_train, df_test = corr_concat(df_train, sp, 90, df_test)

    rs = abs(df_train.corr())
    sp2 = rs.Class[rs.Class < 0.025].index
    df_train = df_train.drop(sp2, axis=1)
    df_test = df_test.drop(sp2, axis=1)

    print(df_train.shape)

    # for i in df_train.columns:
    #     if i != 'Class':
    #         df_train[i] = (df_train[i] - df_train[i].mean())/(df_train[i].max() - df_train[i].min())

    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(ratio=0.78)

    X_resampled, y_resampled = ros.fit_sample(
        df_train.drop(['Class'], axis=1),
        df_train.Class)

    df_resampled = pd.DataFrame(X_resampled, columns=df_train.drop(['Class'], axis=1).columns)
    df_resampled['Class'] = y_resampled

    df_train = df_resampled


    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    svf = LinearSVC(C=400, dual=False)
    svf = LogisticRegression(C=400)
    tm = df_train.Class
    svf = svf.fit(df_train.drop(['Class'], axis=1), df_train.Class)
    df_train = pd.DataFrame(svf.transform(df_train.drop(['Class'], axis=1)))
    df_test = pd.DataFrame(svf.transform(df_test))
    df_train['Class'] = tm
    df_test['Class'] = tm
    print(df_test.shape)

    # model = ExtraTreesClassifier(n_estimators=6000)
    # model.fit(df_train.drop(['Class'], axis=1), df_train.Class)
    # sub = model.predict(df_test.drop(['Class'], axis=1))
    # print(len(sub))
    #
    # pd.DataFrame(sub).to_csv('my_sub.csv', index=False, header=False)
    # exit()

    Lst = [#"from sklearn.ensemble import AdaBoostClassifier",
    #"from sklearn.ensemble import BaggingClassifier",
    "from sklearn.ensemble import ExtraTreesClassifier",
    "from sklearn.ensemble import GradientBoostingClassifier",
    #"from sklearn.ensemble import RandomForestClassifier",
    #"from xgboost import XGBClassifier"
    # "from sklearn.ensemble import VotingClassifier",
    #"from sklearn.gaussian_process import GaussianProcessClassifier",
    # "from sklearn.linear_model import PassiveAggressiveClassifier",
    # "from sklearn.linear_model import RidgeClassifier",
    # "from sklearn.linear_model import RidgeClassifierCV",
    # "from sklearn.linear_model import SGDClassifier",
    # "from sklearn.multiclass import ClassifierMixin",
    # "from sklearn.multiclass import OneVsOneClassifier",
    # "from sklearn.multiclass import OneVsRestClassifier",
    # "from sklearn.multiclass import OutputCodeClassifier",
    # "from sklearn.neighbors import KNeighborsClassifier",
    # "from sklearn.neighbors import RadiusNeighborsClassifier",
    # "from sklearn.neural_network import MLPClassifier",
    # "from sklearn.tree import DecisionTreeClassifier",
    # "from sklearn.tree import ExtraTreeClassifier"
    ]

    sp = []
    glob_dc = {}
    for i in Lst:
        i = i.split('import ')[1]
        try:
            rs = Learning(df_train, y_col='Class').trees({
                'n_estimators':3000,

                }, eval(i), scoring='accuracy')
            print(i, rs)
            if rs > 0.4:
                sp.append(i)
            #fimp = eval(i+' (n_estimators=1000)').fit(df_train.drop(['Class'], axis=1), df_train.Class)
            #glob_dc[i] = dict(zip(df_train.drop(['Class'], axis=1).columns, fimp.feature_importances_))
        except Exception as e:
            print(e)


    print('Mixing...')
    for i in sp:
        locals()['My'+i] = eval(i)()

    print(Learning(df_train, y_col='Class').trees(
            m_params={'demo_param': [eval('My'+i) for i in sp]},
            models=CustomEnsembleClassifier,
            cross=True))