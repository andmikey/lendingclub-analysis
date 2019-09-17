# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd

logger = logging.getLogger(__name__)
    
def add_features(df):
    outlier_columns = ["annual_inc", "revol_bal", "tot_cur_bal", "total_bal_il", "max_bal_bc",
                       "total_rev_hi_lim", "avg_cur_bal", "bc_open_to_buy", "delinq_amnt", "tot_hi_cred_lim",
                       "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit"]
    for col in outlier_columns:
        df = df[df[col] <= df[col].quantile(0.99)]

    df["is_grade_a"] = df["grade"].isin(["A"])
    df["is_grade_a_or_b"] = df["grade"].isin(["A", "B"])
    df["is_grade_f_or_g"] = df["grade"].isin(["F", "G"])
    df["is_verified"] = df["verification_status"] != "Not Verified"
    df["loan:income_ratio"] = df["loan_amnt"] / df["annual_inc"]
    df.drop(columns = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose", "addr_state",
                       "issue_d", "earliest_cr_line", "is_36_month_term"],
            inplace = True)
    return df

def build_features_main(input_file, output_file):
    """ 
    Runs data processing scripts to prepare and clean dataset.
    """
    logger.info('Preparing and cleaning dataset')
    df = pd.read_csv(input_file, low_memory = False)
    logger.info(f"Input dataframe shape: {df.shape}")
    df = add_features(df)
    logger.info(f"Output dataframe shape: {df.shape}")
    logger.info(f"Saving dataframe to {output_file}")
    df.to_csv(output_file, index = False)
    
@click.command()
@click.argument('input_file')
@click.argument('output_file')
def main(input_file, output_file):
    build_features_main(input_file, output_file)
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
