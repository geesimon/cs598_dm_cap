import pickle
import nltk
import json
from tqdm import tqdm
from gensim.summarization.summarizer import summarize

# MAX_DISH_COUNT = 100

with open('./categories/Chinese.pkl', 'rb') as f:
    cuisine_reviews = pickle.load(f)    
with open('ChineseDishes.txt', 'r') as f:
    dishes = {line.replace("\n", "").lower() : {} for line in f.readlines()}
#     dishes = {line.replace("\n", "").lower() : {} for line in f.readlines()[0:MAX_DISH_COUNT]}

for _review in tqdm(cuisine_reviews):
    _review['sentences'] = nltk.sent_tokenize(_review['text'])
    
rest_id2name = {}
path2files="../yelp_dataset_challenge_academic_dataset/"
path2buisness=path2files+"yelp_academic_dataset_business.json"

with open (path2buisness, 'r') as f:
    for line in f.readlines():
        business_json = json.loads(line)
        if (business_json['business_id'] in rest_id2name):
            if rest_id2name[business_json['business_id']] != business_json['name']:
                print('dup name found!')
        else: 
            rest_id2name[business_json['business_id']] = business_json['name']

APP_FOLDER = "../app/data/processed/"
path2files = "../yelp_dataset_challenge_academic_dataset/"
path2buisness = path2files+"yelp_academic_dataset_business.json"

biz = {}
with open (path2buisness, 'r') as f:
    for line in f.readlines():
        business_json = json.loads(line)
        biz[business_json['business_id']] = {}
        for _, key in enumerate(business_json):
            if not key in ['business_id', 'type']:
                biz[business_json['business_id']][key] = business_json[key]

biz2info = {}
for _review in cuisine_reviews:
    if not (_review['business_id'] in biz2info):
        biz2info[_review['business_id']] = {}
        for _, key in enumerate(biz[_review['business_id']]):
            biz2info[_review['business_id']][key] = biz[_review['business_id']][key]
        biz2info[_review['business_id']]['review_summary'] = ''
        biz2info[_review['business_id']]['reviews'] = []
    
    biz2info[_review['business_id']]['reviews'].append({'review_id': _review['review_id'],
                                               'user_id': _review['user_id'], 
                                               'stars': _review['stars'],
                                               'date': _review['date'],
                                               'text': _review['text']
                                              })
    biz2info[_review['business_id']]['review_summary'] += "." + _review['text']

for _id in tqdm(biz2info):
     biz2info[_id]['review_summary'] = summarize(biz2info[_id]['review_summary'], word_count = 120)

with open (APP_FOLDER + 'biz2review.plk', 'wb') as f:
    pickle.dump(biz2info, f)