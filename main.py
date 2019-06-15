import argparse
import pandas as pd
import numpy as np
import os
import re
import io
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
pd.options.mode.chained_assignment = None


def create_corpus(verbose=True, threshold='mean'):

    reports = pd.read_csv('data/reports/reports.csv', encoding='utf-8')
    stats = pd.read_csv('data/stats/stats.csv', encoding='utf-8')

    # Format names for fileids
    reports['name'] = reports['name'].str.replace(' ', '_')
    stats['name'] = stats['name'].str.replace(' ', '_')
    reports['name'] = reports['name'].str.replace('"', '')
    stats['name'] = stats['name'].str.replace('"', '')

    # Shift draft numbers because of new jersey's forfeited pick
    reports_shift = reports[(reports.draft_year == 2011) & (reports.draft_num >= 69)]
    reports_shift['draft_num'] += 1
    reports = reports[~reports['name'].isin(reports_shift['name'])]
    reports = pd.concat([reports, reports_shift])
    reports['word_count'] = reports['report'].str.split().apply(len)

    reports2019 = reports[reports.draft_year == 2019]
    reports_hist = reports[reports.draft_year != 2019]
    merged = pd.merge(reports_hist.drop(columns='name'), stats, on=['draft_year', 'draft_num'], how='inner')

    # Define threshold games played depending on agg function or constant
    if isinstance(threshold, str):
        grouped_stat = merged.groupby('draft_year')['GP'].agg(threshold)
        threshold_gp = merged.apply(lambda x: grouped_stat.loc[x.draft_year, ], axis=1)
        if verbose:
            print(f'================== {threshold.capitalize()} Games Played by Draft Year ==================')
            print(grouped_stat)
    else:
        threshold_gp = threshold
    merged['clas'] = merged['GP'] > threshold_gp

    # Define train and valid set
    mask = (merged['draft_year'] >= 2016) & (merged['clas'] == False)
    reports2019['draft_team'] = 'TBD'
    valid = pd.concat([reports2019, merged[mask][['draft_num', 'draft_year', 'draft_team', 'name', 'report', 'word_count']]], sort=False)
    train = merged[~mask]

    if not os.path.exists('data/merged'):
        os.makedirs('data/merged')
    train.to_csv('data/merged/train.csv')
    valid.to_csv('data/merged/valid.csv')
    if verbose:
        print('Merged data created successfully')

    if not os.path.exists('data/NHLcorpus/true'):
        os.makedirs('data/NHLcorpus/true')
    if not os.path.exists('data/NHLcorpus/false'):
        os.makedirs('data/NHLcorpus/false')
    try:
        for _, row in train.iterrows():
            with io.open(os.path.join('data/NHLcorpus', str(row.clas), '{}_{}.txt'.format(row['name'], row.draft_year)), 'w', encoding='utf-8') as f:
                f.write(row.report)
                f.close()
    except OSError as e:
        print(e)
        return

    if verbose:
        print('Corpus created successfully')


def train_and_predict(verbose=True, show_features=100, save_path='classifiers', long_reports=True):

    # ================================= Read in Data =================================
    train = pd.read_csv('data/merged/train.csv', encoding='utf-8', index_col=0)
    valid = pd.read_csv('data/merged/valid.csv', encoding='utf-8', index_col=0)

    if long_reports:
        train = train[train.apply(lambda x: x.report.count('.'), axis=1) > 1]
        valid = valid[valid.apply(lambda x: x.report.count('.'), axis=1) > 1]

    if verbose:
        print(f'================== Training Distribution ==================')
        print(train['clas'].value_counts())
        print(f'=========== Training Distribution by Draft Year ===========')
        print(train.groupby(['draft_year', 'clas']).size())
        print('============================================================')

    # ================================= Model =================================
    clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(penalty='l2', alpha=1e-3,
                              max_iter=1000, tol=None, loss='log', random_state=123))
    ])
    if verbose:
        print('Training Model...')

    # ================================= Results =================================
    cv_results = cross_validate(clf, train.report, train.clas, cv=10)['test_score']
    clf.fit(train.report, train.clas)
    train_predictions, train_proba = clf.predict(train.report), clf.predict_proba(train.report)
    valid_predictions, valid_proba = clf.predict(valid.report), clf.predict_proba(valid.report)
    train['prediction'], train['confidence'] = train_predictions, train_proba.max(axis=1)
    valid['prediction'], valid['confidence'] = valid_predictions, valid_proba.max(axis=1)

    def adj_proba(row):
        if row.prediction == True:
            return row.confidence
        else:
            return 1 - row.confidence
    train['adj_conf'] = train.apply(lambda row: adj_proba(row), axis=1)
    valid['adj_conf'] = valid.apply(lambda row: adj_proba(row), axis=1)

    vect = clf.get_params()['vect']
    tfidf = clf.get_params()['tfidf']
    clf = clf.get_params()['clf']
    class_labels = clf.classes_
    feature_names = vect.get_feature_names()

    if long_reports:
        dir_name = 'long'
    else:
        dir_name = 'all'
    if not os.path.exists(os.path.join('predictions', dir_name)):
        os.makedirs(os.path.join('predictions', dir_name))
    filepath = os.path.join(save_path, 'Logistic_Regression_{}.pkl'.format(dir_name))
    if verbose:
        print(f'Best cross validation score: {max(cv_results)}')
        print(f'Accuracy on training data: {np.mean(train_predictions == train.clas)}')

        print('================= Classification Report on Train Set =================')
        print(classification_report(train.clas, train.prediction))

        print('================= Erroneous Predictions =================')
        print(train[train.clas != train.prediction].drop(columns='report').head())
        train[train.clas != train.prediction].to_csv(os.path.join('predictions', dir_name, 'erroneous.csv'), index=False)

        for year in valid.draft_year.unique():
            print(f'================= Predictions for {year} =================')
            print(valid[valid.draft_year == year].drop(columns='report').head())
            valid[valid.draft_year == year].to_csv(os.path.join('predictions', dir_name, '{}.csv'.format(year)), index=False)

        print(f'================= Predictions for all Train Set =================')
        print(train.drop(columns='report').head())
        train.to_csv(os.path.join('predictions', dir_name, 'all_train.csv'), index=False)

        print(f'================= Predictions for all Validation Set =================')
        print(valid.drop(columns='report').head())
        valid.to_csv(os.path.join('predictions', dir_name, 'all_valid.csv'), index=False)

        train_true = train[train.clas == True]
        train_true['report'] = train_true['report'] .str.lower()
        train_false = train[train.clas == False]
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
        print('Model Saved')

    else:
        train[train.clas != train.prediction].to_csv(os.path.join('predictions', dir_name, 'erroneous.csv'), index=False)
        for year in valid.draft_year.unique():
            valid[valid.draft_year == year].to_csv(os.path.join('predictions', dir_name, '{}.csv'.format(year)), index=False)
        train.to_csv(os.path.join('predictions', dir_name, 'all_train.csv'), index=False)
        valid.to_csv(os.path.join('predictions', dir_name, 'all_valid.csv'), index=False)
        joblib.dump(clf, filepath)

    # ================================= Plotting =================================
    if not os.path.exists('plots'):
        os.makedirs('plots')
    colors = {1: 'blue', 0: 'red'}
    fig = plt.figure(figsize=(16, 10))
    sns.scatterplot(x='adj_conf', y='GP', data=train, hue='clas', alpha=0.5, palette=colors)
    plt.xticks(np.arange(0, 1.1, step=0.1))
    plt.title("Train Data Confidence vs Games Played")
    plt.xlabel('Model Confidence')
    plt.ylabel('Games Played')
    plt.grid(linestyle='--')
    fig.savefig('plots/Train_Conf_vs_GP.png', bbox_inches='tight')
    if verbose:
        plt.show()

    fig = plt.figure(figsize=(16, 10))
    ax = fig.gca()
    sns.scatterplot(x='adj_conf', y='word_count', data=train, hue='clas', alpha=0.5, palette=colors)
    plt.xticks(np.arange(0, 1.1, step=0.1))
    plt.title("Train Data Confidence vs Word Count")
    plt.xlabel('Model Confidence')
    plt.ylabel('Word Count')
    plt.grid(linestyle='--')
    fig.savefig("plots/Train_Conf_vs_Word_Count.png", bbox_inches='tight')
    if verbose:
        plt.show()

    fig = plt.figure(figsize=(16, 10))
    ax = fig.gca()
    sns.scatterplot(x='word_count', y='GP', data=train, hue='clas', alpha=0.5, palette=colors)
    plt.title("Train Data Word Count vs Games Played")
    plt.xlabel('Word Count')
    plt.ylabel('Games Played')
    plt.grid(linestyle='--')
    plt.savefig("plots/Train_Word_Count_vs_GP.png", bbox_inches='tight')
    if verbose:
        plt.show()


def main(threshold, long_reports, verbose):
    create_corpus(threshold=threshold, verbose=verbose)
    train_and_predict(long_reports=long_reports, verbose=verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define target of model')
    parser.add_argument('-t', '--threshold', dest='threshold', metavar='', type=str,
                        help='String or integer determining threshold of games played to be considered positive class.')
    print_group = parser.add_mutually_exclusive_group()
    print_group.add_argument('-q', '--quiet', action='store_const', dest='verbose', const=False, help='Print quiet')
    print_group.add_argument('-v', '--verbose', action='store_const', dest='verbose', const=True, help='Print verbose')
    report_group = parser.add_mutually_exclusive_group()
    report_group.add_argument('-l', '--long', action='store_const', dest='long', const=True, help='Model on long reports only')
    report_group.add_argument('-a', '--all', action='store_const', dest='long', const=False, help='Model on all reports')
    parser.set_defaults(threshold='mean', verbose=True, long=True)
    args = parser.parse_args()

    if args.threshold.isdigit():
        args.threshold = int(args.threshold)
    main(threshold=args.threshold, long_reports=args.long, verbose=args.verbose)
