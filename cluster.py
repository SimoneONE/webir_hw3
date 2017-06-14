from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt

from sklearn.datasets import load_files

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize

from sklearn.svm import SVC

############################################
stemmer = EnglishStemmer()

def stemming_tokenizer(text):
	stemmed_text = [stemmer.stem(word) for word in word_tokenize(text, language='english')]
	return stemmed_text
######################################################################
def stemming_tokenizer_stopwords_filter(text):
    sw = stopwords.words('english')
    stemmed_text = [stemmer.stem(word) for word in word_tokenize(text, language='english') if word not in sw]
    return stemmed_text
######################################################################
if __name__ == "__main__":
	## Dataset containing Ham and Spam comments
	data_folder_training_set = "./Ham_Spam_comments/Training"

	training_dataset = load_files(data_folder_training_set)

	print("----------------------")
	print("Dataset:")
	print("Total number of training documents: %d" % len(training_dataset.data))
	print("Classes:")
	print(training_dataset.target_names)
	print("----------------------")


	# Load Training-Set
	X_train, X_test_DUMMY_to_ignore, Y_train, Y_test_DUMMY_to_ignore = train_test_split(training_dataset.data,
														training_dataset.target,
														test_size=0.0,
                                                        random_state=42)
	target_names = training_dataset.target_names


	print("----------------------")
	print("Creating Training Set and Test Set")
	print("Training Set Size")
	print(Y_train.shape)
	print("Classes:")
	print(target_names)
	print("----------------------")

	## Vectorization object
	vectorizer = TfidfVectorizer(strip_accents= None,preprocessor = None, tokenizer=stemming_tokenizer_stopwords_filter)
	X = vectorizer.fit_transform(X_train)
	print("n_samples: %d, n_features: %d" % X.shape)
	print(Y_train)
	svd = TruncatedSVD(n_components=2)
	X_svd = svd.fit_transform(X)
	print("n_samples: %d, n_features: %d" % X_svd.shape)
	svc = SVC(kernel="linear", random_state=42,C=10)
	svc.fit(X, Y_train)
    
	Y_pred = svc.predict(X)
        
	X_ham = []
	X_spam = []
	
	for i in range(len(Y_train)):
		if Y_pred[i] == 1:
			X_ham += [X_svd[i]]
		else:
			X_spam += [X_svd[i]]
                



	fig, ax = plt.subplots()
	ax.scatter([a[0] for a in X_ham], [a[1] for a in X_ham], color='b')
	ax.scatter([a[0] for a in X_spam], [a[1] for a in X_spam], color='r')
	plt.show()
