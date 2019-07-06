import pickle
import nltk
import json
from tqdm import tqdm

MAX_DISH_COUNT = 100

with open('./categories/Chinese.pkl', 'rb') as f:
    cuisine_reviews = pickle.load(f)    
with open('ChineseDishes.txt', 'r') as f:
    dishes = {line.replace("\n", "").lower() : {} for line in f.readlines()[0:MAX_DISH_COUNT]}

for _review in tqdm(cuisine_reviews):
    _review['sentences'] = nltk.sent_tokenize(_review['text'].lower())
    
rest2name = {}
path2files="../yelp_dataset_challenge_academic_dataset/"
path2buisness=path2files+"yelp_academic_dataset_business.json"

with open (path2buisness, 'r') as f:
    for line in f.readlines():
        business_json = json.loads(line)
        rest2name[business_json['business_id']] = business_json['name']

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

for _dish in tqdm(dishes):
    dishes[_dish] = {'rest2review_count':{}, 'review2polarity':{}, 'rest2polarity':{}}
    _dish2rest = set()
    _review_count = 0
    _sum_review_polarity = 0
    _sum_rest_polarity = 0
    
    for _review in cuisine_reviews:
        _review_id = _review['review_id']
        _rest_name = rest2name[_review['business_id']]
        _review_polarity = 0
        _review_sentence_count = 0
        for _sentence in _review['sentences']:
            _count = _sentence.count(_dish)
            if _count > 0:
                _review_sentence_count += 1
                _ss = sia.polarity_scores(_sentence)
                _score = (_ss['pos'] + 1e-10)/ (_ss['neg'] + 1e-10) * 0.5
                if _score <= 1:
                    _review_polarity += _score
                else:
                    _review_polarity = 1
                
        
        if _review_sentence_count > 0:
            _review_polarity = _review_polarity / _review_sentence_count
            _sum_review_polarity += _review_polarity
            _review_count += 1
            dishes[_dish]['review2polarity'][_review_id] = _review_polarity
            if _rest_name in dishes[_dish]['rest2review_count']:
                dishes[_dish]['rest2review_count'][_rest_name] += 1
                dishes[_dish]['rest2polarity'][_rest_name] += _review_polarity
            else:
                dishes[_dish]['rest2review_count'][_rest_name] = 1
                dishes[_dish]['rest2polarity'][_rest_name] = _review_polarity
    
            
    for _name in dishes[_dish]['rest2polarity']:
        dishes[_dish]['rest2polarity'][_name] /= dishes[_dish]['rest2review_count'][_name]
        _sum_rest_polarity += dishes[_dish]['rest2polarity'][_name]
    
    dishes[_dish]['rest2polarity'] = sorted(dishes[_dish]['rest2polarity'].items(), key=lambda kv: kv[1], reverse = True)
    dishes[_dish]['rest2review_count'] = sorted(dishes[_dish]['rest2review_count'].items(), key=lambda kv: kv[1], reverse = True)
    dishes[_dish]['review_count'] = len(dishes[_dish]['review2polarity'])
    dishes[_dish]['rest_count'] = len(dishes[_dish]['rest2review_count'])
    dishes[_dish]['rest_polarity'] = _sum_rest_polarity / dishes[_dish]['rest_count']
    dishes[_dish]['review_polarity'] = _sum_review_polarity / dishes[_dish]['review_count']

sorted_rest_dishes = sorted(dishes.items(), key=lambda kv: kv[1]['rest_count'], reverse=True)
sorted_review_dishes = sorted(dishes.items(), key=lambda kv: kv[1]['review_count'], reverse=True)