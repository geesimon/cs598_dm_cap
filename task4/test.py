import pickle
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

MAX_DISH_COUNT = 100

with open('./categories/Chinese.pkl', 'rb') as f:
    cuisine_reviews = pickle.load(f)    
with open('ChineseDishes.txt', 'r') as f:
    dishes = {line.replace("\n", "").lower() : {} for line in f.readlines()[0:MAX_DISH_COUNT]}

for _review in tqdm(cuisine_reviews):
    _review['sentences'] = nltk.sent_tokenize(_review['text'])


sia = SentimentIntensityAnalyzer()

for _dish in tqdm(dishes):
    dishes[_dish] = {'rest2review':{}, 'review2polarity':{}, 'rest2polarity':{}}
    _dish2rest = set()
    _review_count = 0
    
    for _review in cuisine_reviews:
        _review_id = _review['review_id']
        _rest_id = _review['business_id']
        _review_polarity = 0
        _review_sentence_count = 0
        for _sentence in _review['sentences']:
            _count = _sentence.count(_dish)
            if _count > 0:
                _review_sentence_count += 1
                _ss = sia.polarity_scores(_sentence)
                _review_polarity += _ss['compound']
        
        if _review_sentence_count > 0:
            _review_polarity = _review_polarity / _review_sentence_count
            _review_count += 1
            dishes[_dish]['review2polarity'][_review_id] = _review_polarity
            if _rest_id in dishes[_dish]['rest2review']:
                dishes[_dish]['rest2review'][_rest_id].add(_review_id)
                dishes[_dish]['rest2polarity'][_rest_id] += _review_polarity
            else:
                dishes[_dish]['rest2review'][_rest_id] = set([_review_id])
                dishes[_dish]['rest2polarity'][_rest_id] = _review_polarity
    
            
    for _id in dishes[_dish]['rest2polarity']:
        dishes[_dish]['rest2polarity'][_id] /= len(dishes[_dish]['rest2review'][_id])