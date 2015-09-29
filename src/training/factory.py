# -*- coding: utf-8 -*-

from rep.metaml import ClassifiersFactory
from rep.estimators import TMVAClassifier, SklearnClassifier, XGBoostClassifier
from rep.report.metrics import RocAuc
from sklearn.ensemble import AdaBoostClassifier
from copy import deepcopy

# import numpy, pandas
# from rep.utils import train_test_split
# from sklearn.metrics import roc_auc_score
#
# data = pandas.read_csv('toy_datasets/MiniBooNE_PID.txt', sep='\s*', skiprows=[0], header=None, engine='python')
# labels = pandas.read_csv('toy_datasets/MiniBooNE_PID.txt', sep=' ', nrows=1, header=None)
# labels = [1] * labels[1].values[0] + [0] * labels[2].values[0]
# data.columns = ['feature_{}'.format(key) for key in data.columns]
#
# train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.5)

def create_factory(train_variables=["feature_new01: feature_0/feature_1", "feature_2", "feature_26",
                                    "feature_12", "feature_24", "feature_25", "feature_16",]):
    # plot_variables = train_variables + ['feature_3']
    factory = ClassifiersFactory()
    # There are different ways to add classifiers to Factory:
    # factory.add_classifier('tmva', TMVAClassifier(NTrees=50, features=train_variables, Shrinkage=0.05))
    factory.add_classifier('ada', AdaBoostClassifier(n_estimators=10))
    factory['xgb'] = XGBoostClassifier(features=train_variables)
    return factory


def train_factory(factory, train_data, train_labels, train_variables):
    factory.fit(train_data, train_labels, features=train_variables)
    # factory_copy = deepcopy(factory)
    # factory_copy.fit(train_data, train_labels)
    # factory.predict_proba(train_data)

def test_factory(factory, test_data, test_labels):
    report = factory.test_on(test_data, test_labels)
    features_importances = report.feature_importance()
    features_importances.plot()
    learning_curve = report.learning_curve(RocAuc(), metric_label='ROC AUC', steps=1)
    learning_curve.plot(new_plot=True)
    # Plot distribution for each feature
    # use just common features for all classifiers
    report.features_pdf().plot()
    report.prediction_pdf().plot(new_plot=True, figsize = (9, 4))
    # ROC curves (receiver operating characteristic)
    report.roc().plot(xlim=(0.5, 1))