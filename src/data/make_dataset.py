# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
from src.data import clean_dataset
from src.features import build_features
import subprocess

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('n_samples', type=click.INT)
def main(input_filepath, output_filepath, n_samples):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    sample_data = (n_samples > 0)

    fname = 'loan'
    if sample_data:
        logger.info('Sampling data')
        fname = f'loan_sampled_{n_samples}'
        sample_in = os.path.join(input_filepath, "loan.csv")
        sample_out = os.path.join(os.path.dirname(input_filepath), 'interim', f'{fname}.csv')
        logger.info(f'Sampling: {sample_in} -> {sample_out}')
        subprocess.call(['src/data/sample_dataset.sh', str(n_samples), sample_in, sample_out])
        input_filepath = os.path.dirname(sample_out)

    clean_in = os.path.join(input_filepath, f'{fname}.csv')
    clean_out = os.path.join(os.path.dirname(input_filepath), 'interim', f'{fname}-cleaned.csv')
    logger.info(f'Cleaning: {clean_in} -> {clean_out}')
    clean_dataset.clean_dataset_main(clean_in, clean_out)

    features_in = clean_out
    features_out = os.path.join(os.path.dirname(input_filepath), 'processed', f'{fname}.csv')
    logger.info(f'Features: {features_in} -> {features_out}')
    build_features.build_features_main(features_in, features_out)
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
