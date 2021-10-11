# predict.py

import os
import pickle
from sklearn.metrics import accuracy_score
from prep import read_in_data, clean_data
from support import eval_dir, selected_cols, model_dir


def churn_predict():
    '''
    Evaluate the performance of the churn estimator using the evaluation dataset found in the eval_dir/ directory and print the model's accuracy

    INPUT
    None

    OUTPUT
    None
    '''
    df = clean_data(read_in_data(eval_dir), selected_cols)
    y = df.pop('churn')
    X = df

    model_path = os.path.join(model_dir, 'logreg_clf.pkl')
    clf = pickle.load(open(model_path, 'rb'))
    preds = clf.predict(X)

    print('Evaluation accuracy: {:.2f}%'.format(accuracy_score(y, preds) * 100))


if __name__ == '__main__':
    churn_predict()
