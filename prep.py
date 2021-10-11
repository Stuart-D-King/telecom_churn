# prep.py

import re
import numpy as np
import pandas as pd
from support import data_dir


def camel_to_snake(t):
    '''
    Convert strings in camel case to snake case

    https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case

    INPUT
    t: text to convert

    OUTPUT
    s: text converted to snake case
    '''
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', t)
    s = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()
    return s


def read_in_data(data_dir):
    '''
    Read in flat files using Pandas and return a merged dataframe

    INPUT
    data_dir: string of directory name where data lives

    OUTPUT
    df: dataframe of merged data files
    '''
    df_customer = pd.read_csv(data_dir + 'customer_data.csv')
    df_customer.columns = [camel_to_snake(x) for x in df_customer.columns]

    df_internet = pd.read_csv(data_dir + 'internet_data.csv')
    df_internet.columns = [camel_to_snake(x) for x in df_internet.columns]

    df_churn = pd.read_csv(data_dir + 'churn_data.csv')
    df_churn.columns = [camel_to_snake(x) for x in df_churn.columns]

    df = df_customer.merge(df_internet, on='customer_id').merge(df_churn, on='customer_id')
    df.drop('customer_id', axis=1, inplace=True)

    return df


def clean_data(df, cols):
    '''
    Clean data by converting boolean variables to 1/0 and creating one-hot-encoded representations of categorical data

    INPUT
    df: dataframe to clean
    cols: list of column names for returned dataframe

    OUTPUT
    df: cleaned dataframe
    '''
    boolean_cols = ['churn', 'senior_citizen', 'partner', 'dependents', 'online_security', 'tech_support', 'paperless_billing']
    for b in boolean_cols:
        if b in df.columns:
            df[b] = np.where((df[b] == 'Yes') | (df[b] == 1), 1, 0)

    categorical_cols = ['internet_service', 'contract', 'payment_method']
    dummies = pd.get_dummies(df[categorical_cols])
    dummies.columns = [c.lower().replace(' ', '_').replace('-', '_').replace(')', '').replace('(', '') for c in dummies.columns]

    df = pd.concat([df, dummies], axis=1)
    df = df[cols]

    return df


def main():
    return None


if __name__ == '__main__':
    main()
