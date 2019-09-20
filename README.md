lendingclub-analysis
==============================

This repository details an initial approach to developing a predictive model for the Lending Club loan data, available [here](https://www.kaggle.com/wendykan/lending-club-loan-data/downloads/lending-club-loan-data.zip).

In this project, we develop a model to detect if a loan will default before the loan is issued. Headline figures for the tested models on the full dataset:

|  Metric   | Logistic Regression | Decision Tree | Random Forest | **Gaussian** |
|-----------|---------------------|---------------|---------------|--------------|
| AUC       | 0.72                | 0.52          | 0.59          | 0.64         |
| Accuracy  | 0.41                | 0.50          | 0.61          | 0.71         |
| Precision | 0.94                | 0.80          | 0.82          | 0.82         |
| Recall    | 0.27                | 0.48          | 0.63          | 0.82         |


Instructions 
------------

1. Clone the repository:

```bash
$ git clone https://github.com/andmikey/lendingclub-analysis.git
```

2. Download the data and place in correct locations:

```
This is best done by hand, unless you already have the API tool set up.
Go to the link above, download the zip file to lending-club-loan-data.zip, unzip it, and move the files to their appropriate directories:
$ mkdir data
$ mkdir data/raw
$ mkdir data/interim
$ mkdir data/processed
$ mv loan.csv data/raw/
$ mv LCDataDictionary.xlsx references/
```

3. Start a new pip environment and install the required packages:

```bash
$ virtualenv -p `which python3` env
$ source env/bin/activate
$ pip3 install -r requirements.txt
```

4. Pre-process the dataset for training:

```bash
$ make data N_SAMPLES=0
```

5. Train the model:

```bash
$ make train_model DATA=data/processed/loan.csv MODEL=models/model.pickle
```

6. (Optional) If there is a new unlabelled dataset, say at data/processed/new_data.csv, predict the labels for the new observations:

```bash
$ make predict_model MODEL=models/model.pickle INPUT=data/processed/new_data.csv OUTPUT=models/test_predict.csv
```

7. (Optional) You can also do piece-wise data creation: sampling the data (to reduce training size), cleaning the data, and adding features:

```bash
$ make sample_data N_SAMPLES=50000
$ make clean_data SRC=data/interim/loan_sampled_50000.csv DEST=data/interim/loan_sampled_50000-cleaned.csv
$ make add_features SRC=data/interim/loan_sampled_50000-cleaned.csv DEST=data/processed/loan_sampled_50000.csv
```

Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a date, username, and description.
    │   └── exploratory    <- Notebooks for exploring and testing approaches (cleaning, visualisation, prediction)
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis 
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to generate data
    │   │   ├── clean_dataset.py
    │   │   ├── make_dataset.py    
    │   │   └── sample_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
