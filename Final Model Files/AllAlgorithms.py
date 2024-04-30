import pandas as pd 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import tensorflow as tf 

from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam 

from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, minmax_scale

from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from scipy.ndimage import convolve


import collections 
import random

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(algorithm="SAMME", random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

def makeDBNNetwork(num_layers, **kwargs):
    # 1 Layer

    current_pipe = Pipeline([
        ('rbm1', BernoulliRBM(n_components=kwargs["L1_COMPONENTS"], n_iter = kwargs["L1_ITER"]))
    ])

    current_pipe.add

X_train = []

for _name, _clf in zip(names, classifiers):

    # 1 - Layer Pipeline
    for multip in [0.5, 1, 2, 4]:
        L1_PIPELINE = Pipeline([
            ('mmx1', MinMaxScaler()),
            ('rbm1', BernoulliRBM(n_components = multip * X_train.shape[2], n_iter = 10)),
            (_name, _clf)
        ])

        print(L1_PIPELINE)

    # 2 - Layer Pipeline


    # 3 - Layer Pipeline

    # 4 - Layer Pipeline

    # 5 - Layer Pipeline

    print(_name," ", _clf)