import nltk
from nltk.corpus import CategorizedPlaintextCorpusReader,stopwords
import random
import pickle
import re

def display_features(num_features=1000,show_features=200,fileid='classifiers/nltk_nb_filter.pickle'):
	'''
	Displays informative features from NHLCorpus
	'''
	stop_words = set(stopwords.words('english'))
	nhl = CategorizedPlaintextCorpusReader(root='data/NHLcorpus/',fileids=r'.*\.txt',cat_pattern='(\w+)/*')
	documents = [([re.sub(r'\W+','',w.lower()) for  w in nhl.words(fileid) if w.lower() not in stop_words],category)
				for category in nhl.categories()
				for fileid in nhl.fileids(category)]
	all_words = nltk.FreqDist(re.sub(r'\W+','',w.lower()) for w in nhl.words() if w.lower() not in stop_words)
	word_features = [w[0] for w in all_words.most_common(num_features)]

	def document_features(document):
		document_words = set(document)
		features = {}
		for word in word_features:
			features['contains({})'.format(word)] = word in document_words
		return features

	featuresets = [(document_features(d),c) for (d,c) in documents]
	nb_clf = nltk.NaiveBayesClassifier.train(featuresets)
	nb_clf.show_most_informative_features(show_features)
	print('Accuracy on training data: {}'.format(nltk.classify.accuracy(nb_clf,featuresets)))

	save_classifier = open(fileid,'wb')
	pickle.dump(nb_clf,save_classifier)
	save_classifier.close()

if __name__ == '__main__':
	display_features()


