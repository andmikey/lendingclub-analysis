# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import subprocess
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def normalize_df(df):
    """
    Normalize all numeric columns in the dataframe to be between 0 and 1. 
    """
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df.values), 
                       columns=df.columns, index=df.index)
    return df

@click.command()
@click.argument('model_filepath', type=click.Path(exists = True))
@click.argument('data_to_predict', type=click.Path(exists = True))
@click.argument('predictions_output', type=click.Path())
def main(model_filepath, data_to_predict, predictions_output):
    """ 
    Predicts class of new observations using trained model, and saves predictions to predictions_output. 

    Parameters:
    model_filepath (string): Filepath of trained model. 
    data_to_predict (string): Filepath of new observations. 
    predictions_output (string): Filepath to save predictions. 

    Side effects:
    Saves predictions to predictions_output. 
    """
    logger.info(f"Predicting file {data_to_predict} from model at {model_filepath}")
    model = pickle.load(open(model_filepath, 'rb'))
    df_to_predict = pd.read_csv(data_to_predict, low_memory = False)
    normalized = normalize_df(df_to_predict)
    predictions = model.predict(normalize_df(df_to_predict))
    logger.info(f"Saving predictions to {predictions_output}")
    np.savetxt(predictions_output, predictions, delimiter = ',', fmt="%i")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
