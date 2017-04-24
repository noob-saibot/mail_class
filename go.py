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
    df_train = pd.read_csv('x_train.csv', delimiter=';', header=None)
    df_test = pd.read_csv('x_test.csv', delimiter=';', header=None)
    df_sub = pd.read_csv('y_train.csv', delimiter=';', names=['Class'])
    df_rs = pd.concat([df_train, df_sub], axis=1)
    df_rs = pd.concat([df_rs, df_test], ignore_index=True)
    del df_test, df_train, df_sub

    df_train = df_rs.dropna()

    dct_params = {
        'verbose': True,
        'criterion': 'mse',
        'max_features': 'log2',
        'random_state': 1234,
        'subsample': 0.9,
        'max_depth': 5,
        # 'n_estimators': 1000,
        # 'presort': False,


    }

    # for i in dir(sklearn):
    #     for j in eval('dir(sklearn.%s)'%i):
    #         if 'Classifier' in j:
    #             print('"from sklearn.', i, ' import ', j, '",', sep='')

    Lst = ["from sklearn.base import ClassifierMixin",
"from sklearn.ensemble import AdaBoostClassifier",
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
    #df_train = df_train.head(10)
    for i in Lst:
        i = i.split('import ')[1]
        try:
            rs = Learning(df_train, y_col='Class').trees({}, eval(i), scoring='accuracy')
            print(i, rs)
            if rs > 0.48:
                sp.append(i)
        except:
            pass

    for i in sp:
        locals()['My'+i] = eval(i)()

    print(Learning(df_train, y_col='Class').trees(
        m_params={'demo_param': [eval('My'+i) for i in sp]},
        models=CustomEnsembleClassifier,
        cross=True))

