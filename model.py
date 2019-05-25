import pandas as pd
import numpy as np
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def main(verbose=True, show_features=100, filepath='classifiers/SVM.pkl'):

    # ================================= Read in Data =================================
    train = pd.read_csv('data/merged/train.csv', encoding='utf-8', index_col=0)
    valid = pd.read_csv('data/merged/valid.csv', encoding='utf-8', index_col=0)

    # ================================= Model =================================
    clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(penalty='l2', alpha=1e-3,
                              max_iter=5, tol=None, loss='log', random_state=123))
    ])

    # ================================= Results =================================
    cv_results = cross_validate(clf, train.report, train.NHL, cv=10)['test_score']
    clf.fit(train.report, train.NHL)
    train_predictions, train_proba = clf.predict(train.report), clf.predict_proba(train.report)
    valid_predictions, valid_proba = clf.predict(valid.report), clf.predict_proba(valid.report)
    train['prediction'], train['probability'] = train_predictions, train_proba.max(axis=1)
    valid['prediction'], valid['probability'] = valid_predictions, valid_proba.max(axis=1)
    vect = clf.get_params()['vect']
    tfidf = clf.get_params()['tfidf']
    clf = clf.get_params()['clf']
    class_labels = clf.classes_
    feature_names = vect.get_feature_names()
    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    if verbose:
        print(f'Best cross validation score: {max(cv_results)}')
        print(f'Accuracy on training data: {np.mean(train_predictions == train.NHL)}')

        print('================= Classification Report on Train Set =================')
        print(classification_report(train.NHL, train.prediction))

        print('================= Erroneous Predictions =================')
        print(train[train.NHL != train.prediction].drop(columns='report'))
        train[train.NHL != train.prediction].to_csv('predictions/erroneous.csv')

        for year in valid.draft_year.unique():
            print(f'================= Predictions for {year} =================')
            print(valid[valid.draft_year == year].drop(columns='report'))
            valid[valid.draft_year == year].to_csv(f'predictions/{year}.csv')

        print(f'================= Predictions for all Train Set =================')
        print(train.drop(columns='report'))
        train.to_csv('predictions/all_train.csv')

        print(f'================= Predictions for all Validation Set =================')
        print(valid.drop(columns='report'))
        valid.to_csv('predictions/all_valid.csv')

        train_true = train[train.NHL == True]
        train_true['report'] = train_true['report'] .str.lower()
        train_false = train[train.NHL == False]
        train_false['report'] = train_false['report'] .str.lower()

        print('================= Important Positive Features =================')
        for i, j in enumerate(clf.coef_[0].argsort()[-show_features:][::-1]):
            feature = feature_names[j]
            true_count = train_true['report'].str.contains(feature).sum()
            false_count = train_false['report'].str.contains(feature).sum()
            print('{0:3}:{1:20} T:{2:3} F:{3:3}'.format(i + 1, feature, true_count, false_count))

        print('================= Important Negative Features =================')
        for i, j in enumerate(clf.coef_[0].argsort()[:show_features]):
            feature = feature_names[j]
            true_count = train_true['report'].str.contains(feature).sum()
            false_count = train_false['report'].str.contains(feature).sum()
            print('{0:3}:{1:20} T:{2:3} F:{3:3}'.format(i + 1, feature, true_count, false_count))

        joblib.dump(clf, filepath)
        if verbose:
            print('Model Saved')


if __name__ == '__main__':
    main()
