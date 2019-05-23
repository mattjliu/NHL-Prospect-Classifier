import pandas as pd
import numpy as np
import os
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def main(verbose=True):
	reports = pd.read_csv('data/reports/reports_v2.csv',encoding='utf-8')
	stats = pd.read_csv('data/stats/stats_v2.csv',encoding='utf-8')

	# Format names for fileids
	reports['name'] = reports['name'].str.replace(' ','_')
	stats['name'] = stats['name'].str.replace(' ','_')
	reports['name'] = reports['name'].str.replace('"','')
	stats['name'] = stats['name'].str.replace('"','')

	# Shift draft numbers because of new jersey's forfeited pick
	reports_shift = reports[(reports.draft_year == 2011) & (reports.draft_num >= 69)]
	reports_shift['draft_num'] += 1
	reports = reports[~reports['name'].isin(reports_shift['name'])]
	reports = pd.concat([reports,reports_shift])

	reports2019 = reports[reports.draft_year == 2019]
	reports_hist = reports[reports.draft_year != 2019]
	merged = pd.merge(reports_hist.drop(columns='name'),stats,on=['draft_year','draft_num'],how='inner')
	merged['NHL'] = merged['GP'] > 0

	# Define train set
	mask = (merged['draft_year'] >= 2016) & (merged['NHL'] == False)
	valid = pd.concat([reports2019,merged[mask][['draft_num','draft_year','name','report']]])
	train = merged[~mask]

	# ================================= Model and optimization =================================
	clf = Pipeline([
	('vect',CountVectorizer()),
	('tfidf',TfidfTransformer()),
	('clf',SGDClassifier(penalty='l2',alpha=1e-3,
						random_state=1,max_iter=5,tol=1e-3))
	])

	params = {
	'vect__ngram_range':[(1,1),(1,2)],
	'tfidf__use_idf':(True,False),
	'clf__alpha':(1e-2,1e-3),
	'clf__loss':('hinge','log'),
	'clf__penalty':('l1','l2','elasticnet'),
	}

	if verbose: 
		print('Training Model...')
	gs_clf = GridSearchCV(clf,params,cv=5,iid=False,n_jobs=-1)
	gs_clf.fit(train.report,train.NHL)
	best_clf = gs_clf.best_estimator_
	if verbose:
		print('================= Best Parameters =================')
		print(gs_clf.best_params_)
		print('================= Steps =================')
		print(best_clf.steps)

	# ================================= Results =================================
	train_predictions = best_clf.fit(train.report,train.NHL).predict(train.report)
	valid_predictions = best_clf.fit(train.report,train.NHL).predict(valid.report)
	train['predictions'] = train_predictions
	valid['NHL'] = valid_predictions
	vect = best_clf.get_params()['vect']
	tfidf = best_clf.get_params()['tfidf']
	clf = best_clf.get_params()['clf']
	class_labels = clf.classes_
	feature_names = vect.get_feature_names()
	if not os.path.exists('predictions'):
		os.makedirs('predictions')
	if verbose:
		print(f'Accuracy on training data: {np.mean(train_predictions == train.NHL)}')
		print('================= Erroneous Predictions =================')
		print(train[train.NHL != train.predictions].drop('report',axis=1))
		train[train.NHL != train.predictions].to_csv('predictions/erroneous.csv')
		for year in valid.draft_year.unique():
			print(f'================= Predictions for {year} =================')
			print(valid[valid.draft_year == year])
			valid[valid.draft_year == year].to_csv(f'predictions/{year}.csv')
		print('================= Important Positive Features =================')
		for i,j in enumerate(clf.coef_[0].argsort()[-100:][::-1]):
			print(f'{i+1}:{feature_names[j]}')
		print('================= Important Negative Features =================')
		for i,j in enumerate(clf.coef_[0].argsort()[:100]):
			print(f'{i+1}:{feature_names[j]}')

if __name__ == '__main__':
	main()