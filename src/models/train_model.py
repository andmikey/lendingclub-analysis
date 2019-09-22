# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import subprocess
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd

logger = logging.getLogger(__name__)

RANDOM_STATE = 10

def data_from_dataset(filename):
    """
    Reads data from CSV and split to train/test. 
    """
    df = pd.read_csv(filename, 
                low_memory = False)
    
    y = (df["target"]).astype(int)
    X = df.drop(columns = ["target"])
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, y, 
                                                                  random_state = RANDOM_STATE,
                                                                  train_size = 0.9)
    logger.info(f"Train size: {X_train_orig.shape}\nTest size: {X_test_orig.shape}")
    return X_train_orig, X_test_orig, y_train, y_test

def normalize_df(df):
    """
    Normalize numerical colums in dataframe to be between 0 and 1. 
    """
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df.values), 
                       columns=df.columns, index=df.index)
    return df

def undersample_dataset(X_train_orig, y_train):
    """
    Undersamples the dataset for class 0 so there is an equal amount of 0 and 1 class entries. 
    """
    df_train = pd.concat([X_train_orig, y_train], axis = 1)
    count_class_0, count_class_1 = df_train.target.value_counts()
    df_class_0 = df_train[df_train['target'] == 0]
    df_class_1 = df_train[df_train['target'] == 1]
    df_class_0_under = df_class_0.sample(count_class_1)
    df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

    logger.info(df_test_under.target.value_counts())
    X_train_under = df_test_under.drop(columns=["target"])
    y_train_under = df_test_under["target"].astype(int)

    return X_train_under, y_train_under

def calc_metrics(conf_mat):
    """
    Calculates metrics based on confusion matrix. 
    """
    tn, fp, fn, tp = conf_mat.ravel()
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall}

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ 
    Trains and saves model. 
    
    Parameters:
    input_filepath (string): Location of data to use to train and test the model.
    output_filepath (string): Location to save trained model. 
    """
    logger.info(f"Training model on data at {input_filepath}")
    
    (X_train_orig, X_test_orig, y_train, y_test) = data_from_dataset(input_filepath)
    X_train_orig.fillna(0, inplace = True)
    X_test_orig.fillna(0, inplace = True)
    (X_train_under, y_train_under) = undersample_dataset(X_train_orig, y_train)
    X_train = normalize_df(X_train_under)
    X_test = normalize_df(X_test_orig)
    
    model = GaussianNB()
    model.fit(X_train, y_train_under)

    y_preds = model.predict(X_test)
    y_preds_probs = model.predict_proba(X_test)
    
    auc = roc_auc_score(y_test, y_preds_probs[:, 1])
    conf_mat = confusion_matrix(y_test, y_preds, [1, 0])
    metrics = calc_metrics(conf_mat)
    logger.info(f"Accuracy : {metrics['accuracy']:.2f}")
    logger.info(f"Precision: {metrics['precision']:.2f}")
    logger.info(f"Recall   : {metrics['recall']:.2f}")
    logger.info(f"Confusion matrix: \n{conf_mat}")

    logger.info(f"Saving model to {output_filepath}")
    pickle.dump(model, open(output_filepath, 'wb'))
    
    return

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
