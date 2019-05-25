import pandas as pd
import numpy as np
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def main(verbose=True,show_features=100,filepath='classifiers/SVM.pkl'):
	
	# ================================= Read in Data =================================
	train = pd.read_csv('data/merged/train.csv',encoding='utf-8')
	valid = pd.read_csv('data/merged/valid.csv',encoding='utf-8')

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
		print()

	# ================================= Results =================================
	best_clf.fit(train.report,train.NHL)
	train_predictions,train_proba = best_clf.predict(train.report),best_clf.predict_proba(train.report)
	valid_predictions,valid_proba = best_clf.predict(valid.report),best_clf.predict_proba(valid.report)
	train['prediction'],train['probability'] = train_predictions,train_proba.max(axis=1)
	valid['prediction'],valid['probability'] = valid_predictions,valid_proba.max(axis=1)
	vect = best_clf.get_params()['vect']
	tfidf = best_clf.get_params()['tfidf']
	clf = best_clf.get_params()['clf']
	class_labels = clf.classes_
	feature_names = vect.get_feature_names()
	if not os.path.exists('predictions'):
		os.makedirs('predictions')
	if verbose:
		print(f'Best cross validation score: {gs_clf.best_score_}')
		print(f'Accuracy on training data: {np.mean(train_predictions == train.NHL)}')

		print('================= Classification Report on Train Set =================')
		print(classification_report(train.NHL,train.prediction))

		print('================= Erroneous Predictions =================')
		print(train[train.NHL != train.prediction].drop(columns='report'))
		train[train.NHL != train.prediction].to_csv('predictions/erroneous.csv')

		for year in valid.draft_year.unique():
			print(f'================= Predictions for {year} =================')
			print(valid[valid.draft_year == year])
			valid[valid.draft_year == year].to_csv(f'predictions/{year}.csv')

		print(f'================= Predictions for all Train Set =================')
		print(train.drop(columns='report'))
		train.to_csv('predictions/all_train.csv')

		print(f'================= Predictions for all Validation Set =================')
		print(valid)
		valid.to_csv('predictions/all_valid.csv')

		train_true = train[train.NHL == True]
		train_true['report'] = train_true['report'] .str.lower()
		train_false = train[train.NHL == False]
		train_false['report']  = train_false['report'] .str.lower()

		print('================= Important Positive Features =================')
		for i,j in enumerate(clf.coef_[0].argsort()[-show_features:][::-1]):
			feature = feature_names[j]
			true_count = train_true['report'].str.contains(feature).sum()
			false_count = train_false['report'].str.contains(feature).sum()
			print('{0:3}:{1:20} T:{2:3} F:{3:3}'.format(i+1,feature,true_count,false_count))

		print('================= Important Negative Features =================')
		for i,j in enumerate(clf.coef_[0].argsort()[:show_features]):
			feature = feature_names[j]
			true_count = train_true['report'].str.contains(feature).sum()
			false_count = train_false['report'].str.contains(feature).sum()
			print('{0:3}:{1:20} T:{2:3} F:{3:3}'.format(i+1,feature,true_count,false_count))

		joblib.dump(best_clf,filepath)

if __name__ == '__main__':
	main()