from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from tqdm import tqdm
import numpy as np
import pandas as pd
import string
import time


stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    result_stemmed = []
    for token in simple_preprocess(text, min_len = 2):
        result.append(token)
#         if token not in STOPWORDS:
        result_stemmed.append(lemmatize_stemming(token))
    
    return (result, result_stemmed)

with open("./data/raw/Hygiene/hygiene.dat.labels") as f:
    LABELS = [int(l) for l in f.readlines() if l[0].isdigit()]

ALL_RAW_TEXTS = []
ALL_TEXTS = []
ALL_STEMMED_TEXTS = []
ALL_CONCAT_STEMMED_TEXTS = []
LABELED_TEXTS = []
LABELED_CONCAT_TEXTS = []
LABELED_STEMMED_TEXTS = []
LABELED_CONCAT_STEMMED_TEXTS = []

with open("./data/raw/Hygiene/hygiene.dat") as f:
    ALL_RAW_TEXTS = f.readlines()

for _text in tqdm(ALL_RAW_TEXTS[0:1000]):
    _result, _result_stemmed = preprocess(_text)
    ALL_TEXTS.append(_result)
    ALL_STEMMED_TEXTS.append(_result_stemmed)

ALL_CONCAT_STEMMED_TEXTS = [" ".join(_text) for _text in ALL_STEMMED_TEXTS]

LABELED_TEXTS = ALL_TEXTS[0:len(LABELS)]
LABELED_CONCAT_TEXTS = [" ".join(_text) for _text in LABELED_TEXTS]

LABELED_STEMMED_TEXTS = ALL_STEMMED_TEXTS[0:len(LABELS)]
LABELED_CONCAT_STEMMED_TEXTS = [" ".join(_text) for _text in LABELED_STEMMED_TEXTS]

FEATURE_MORE =pd.read_csv("./data/raw/Hygiene/hygiene.dat.additional", header=None)

# create a dataframe using texts and lables
labeled_df = pd.DataFrame()
labeled_df['concat_stemmed_text'] = LABELED_CONCAT_STEMMED_TEXTS
labeled_df['stemmed_text'] = LABELED_STEMMED_TEXTS
# labeled_df['concat_stemmed_text'] = LABELED_CONCAT_TEXTS
# labeled_df['stemmed_text'] = LABELED_TEXTS
labeled_df['label'] = LABELS

# split the dataset into training and validation datasets 
train_concat_stemmed_text, test_concat_stemmed_text, train_label, test_label = model_selection.train_test_split(labeled_df['concat_stemmed_text'], 
                                                                                  labeled_df['label'],
                                                                                  test_size = 0.2)
train_stemmed_text = labeled_df['stemmed_text'][train_concat_stemmed_text.index]
test_stemmed_text = labeled_df['stemmed_text'][test_concat_stemmed_text.index]

# # label encode the target variable 
# encoder = preprocessing.LabelEncoder()
# train_label = encoder.fit_transform(train_label)
# test_label = encoder.fit_transform(test_label)


class MyCountVectorizer:
    def __init__(self, max_df = 0.5):
        self.vect_model = CountVectorizer(analyzer='word', max_df = max_df)
        
    def fit(self, texts_concat):
        self.vect_model.fit(texts_concat)
    
    def transform(self, texts_concat):
        return self.vect_model.transform(texts_concat)

count_vect = MyCountVectorizer()
count_vect.fit(train_concat_stemmed_text)
train_count = count_vect.transform(train_concat_stemmed_text)
test_count =  count_vect.transform(test_concat_stemmed_text)

class MyTfidfVectorizer(MyCountVectorizer):
    def __init__(self, analyzer='word', ngram_range = None, max_features=5000):
        if ngram_range is None:
            self.vect_model = TfidfVectorizer(analyzer = analyzer, max_features = max_features)
        else:
            self.vect_model = TfidfVectorizer(analyzer = analyzer, ngram_range = ngram_range, 
                                              max_features = max_features)
            
    def fit(self, texts_concat):   
        self.vect_model.fit(texts_concat)
        self.vocabulary = self.vect_model.vocabulary_


tfidf_vect = MyTfidfVectorizer()
tfidf_vect.fit(train_concat_stemmed_text)
train_tfidf =  tfidf_vect.transform(train_concat_stemmed_text)
test_tfidf =  tfidf_vect.transform(test_concat_stemmed_text)

# ngram level tf-idf 
tfidf_vect_ngram = MyTfidfVectorizer(ngram_range=(2,3))
tfidf_vect_ngram.fit(train_concat_stemmed_text)
train_tfidf_ngram =  tfidf_vect_ngram.transform(train_concat_stemmed_text)
test_tfidf_ngram =  tfidf_vect_ngram.transform(test_concat_stemmed_text)

# characters level tf-idf
tfidf_vect_ngram_chars = MyTfidfVectorizer(analyzer='char', ngram_range=(2,3))
tfidf_vect_ngram_chars.fit(train_concat_stemmed_text)
train_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_concat_stemmed_text) 
test_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(test_concat_stemmed_text) 


from sklearn import linear_model, naive_bayes, metrics, svm

def show_score(classifier_name, scores):
    print("Accuracy:%0.2f Precission:%0.2f Recall:%0.2f F1:%0.2f"%scores, "-> [%s]"%(classifier_name))
    
def train_model(classifier, train_feature, train_label, test_feature, test_label, is_neural_net=False):
#     print(train_feature)
    # fit the training dataset on the classifier
    classifier.fit(train_feature, train_label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(test_feature)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return (metrics.accuracy_score(predictions, test_label),
            metrics.precision_score(predictions, test_label),
            metrics.recall_score(predictions, test_label),
            metrics.f1_score(predictions, test_label))

def run_model(classfier_configs):
    for _name in classfier_configs:
#         print(classfier_configs[_name].train)
        scores = train_model(classfier_configs[_name][0], classfier_configs[_name][1], train_label, classfier_configs[_name][2], test_label)
        show_score(_name, scores)


# nb_classifiers = {
#     "NB Count": (naive_bayes.MultinomialNB(), train_count, test_count)
#     , "NB TFIDF": (naive_bayes.MultinomialNB(), train_tfidf, test_tfidf)
#     , "NB TFIDF NGram": (naive_bayes.MultinomialNB(), train_tfidf_ngram, test_tfidf_ngram)
#     , "NB TFIDF NGram Chars": (naive_bayes.MultinomialNB(), train_tfidf_ngram_chars, test_tfidf_ngram_chars)
#     , # "NB LDA": (naive_bayes.MultinomialNB(), train_lda, test_lda),
# }

# run_model(nb_classifiers)

svm_classifiers = {
    "SVM Count": (svm.SVC(), train_count, test_count)
    ,"SVM TFIDF": (svm.SVC(), train_tfidf, test_tfidf)
    ,"SVM TFIDF NGram": (svm.SVC(), train_tfidf_ngram, test_tfidf_ngram)
    ,"SVM TFIDF NGram Chars": (svm.SVC(), train_tfidf_ngram_chars, test_tfidf_ngram_chars),
}
    
run_model(svm_classifiers)
