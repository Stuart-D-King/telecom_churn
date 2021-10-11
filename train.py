# train.py

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from prep import read_in_data, clean_data
from support import logreg_params, data_dir, model_dir, selected_cols


def run_gridsearch(model, param_grid, X, y, scoring='accuracy', cv=5, n_jobs=1):
    '''
    Perform a hyperparameter grid search for the passed in model and parameter grid and display output

    INPUT
    model: model to tune
    param_grid: dictionaruy defining the grid search parameters
    X: feature dataset
    y: target labels
    scoring: type of scoring metric
    cv: number of cross validation folds (default=5)
    n_jobs: number of jobs to runin parallel (default=1)

    OUTPUT
    best_model: best estimator refit on all training data
    '''
    gs = GridSearchCV(
        model,
        param_grid,
        n_jobs=n_jobs,
        scoring=scoring,
        cv=cv
    )

    gs.fit(X,y)
    best_params = gs.best_params_
    best_model = gs.best_estimator_

    print('\nResults of Gridsearch:')
    print('{0:<20s} | {1:<10s} | {2}'.format('Parameter', 'Optimal', 'GS Values'))
    print('-' * 80)
    for param, vals in param_grid.items():
        print('{0:<20s} | {1:<10s} | {2}'.format(str(param), str(best_params[param]), str(vals)))

    return best_model


def train_classifier():
    '''
    Trains a logistic regression model for telecom churn prediction and displays a classification report for the test holdout set

    INPUT
    None

    OUTPUT
    logreg_best: best logistic regressor based on hyperparameter tuning
    '''
    # get the data
    df = clean_data(read_in_data(data_dir), selected_cols)

    y = df.pop('churn')
    X = df

    numeric_features = ['tenure', 'monthly_charges']
    boolean_features = [c for c in X.columns if c not in numeric_features]

    # split data into training and test sets, stratify by target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

    # build a preprocessor transformer of boolean and numeric pipeline transformers
    boolean_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('bool', boolean_transformer, boolean_features),
            ('num', numeric_transformer, numeric_features)
        ])

    logreg = LogisticRegression(random_state=1)
    logreg_pipeline = Pipeline(steps=[
        ('p', preprocessor),
        ('m', logreg)
    ])

    # perform cross validation to evaluate model performance for out of the box Logistic Regressor
    logreg_scores = cross_val_score(logreg_pipeline, X_train, y_train, scoring='accuracy')
    print('Logistic Regression mean CV accuracy: {:2f}%'.format(np.mean(logreg_scores) * 100))

    # run a grid search for the logistic regression algorithm
    logreg_best = run_gridsearch(
        logreg_pipeline,
        logreg_params,
        X_train,
        y_train
    )

    # evaluate the final tuned model using the holdout test set
    print('\n')
    print(classification_report(y_test, logreg_best.predict(X_test)))

    clf_path = os.path.join(model_dir, 'logreg_clf.pkl')

    # save the fit model as a pickle file
    with open(clf_path, 'wb') as handle:
        pickle.dump(logreg_best, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    train_classifier()
