import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer


def load_breast_cancer_dataset():
    breast_cancer = load_breast_cancer()
    scaler = MinMaxScaler()
    scaler.fit_transform(breast_cancer.data)
    return breast_cancer.data, breast_cancer.target


def load_data(dataset_name, model = None, train_size = None):
    if dataset_name == 'adult':
        X, y = load_adult()
    elif dataset_name == 'digits':
        X, y = load_digits()
    elif dataset_name == 'breast_cancer':
        X, y = load_breast_cancer_dataset()
    return X,y

def load_adult():
    df = pd.read_csv("adult/adult.data")
    df.columns = ["age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "gender",
    "capital-gain", "capital-loss", "hpw", "country", "income"]

    #change income to 0/1
    income_map={' <=50K':0,' >50K':1}
    df['income']=df['income'].map(income_map).astype(int)

    #drop capital gain and loss as mostly 0s
    df.drop("capital-gain", axis=1,  inplace=True)
    df.drop("capital-loss", axis=1,  inplace=True)

    #save target variable and drop from dataframe
    y=df['income']
    df.drop("income", axis=1,  inplace=True)

    #convert target variable to dummies
    categorical_columns=["workclass", "education",
    "marital-status", "occupation", "relationship", "race", "gender", "country"]
    df = pd.get_dummies(df, columns=categorical_columns)

    scaler = MinMaxScaler()
    scale_cols=["age", "fnlwgt", "education-num", "hpw"]
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    X=df.to_numpy()
    y=y.to_numpy()

    return X, y

def load_digits():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target
