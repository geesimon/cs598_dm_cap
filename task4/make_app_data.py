import pickle
import json
from tqdm import tqdm
from gensim.summarization.summarizer import summarize
from pyteaser import Summarize

with open('./categories_2/Chinese.pkl', 'rb') as f:
    cuisine_reviews = pickle.load(f)

APP_FOLDER = "../app/data/processed/"
path2files = "../yelp_dataset_challenge_academic_dataset/"
path2buisness = path2files+"yelp_academic_dataset_business.json"

rest = {}
with open(path2buisness, 'r') as f:
    for line in f.readlines():
            business_json = json.loads(line)
            rest[business_json['business_id']] = {}
            for _, key in enumerate(business_json):
                if not key in ['business_id', 'type']:
                    rest[business_json['business_id']][key] = business_json[key]

rest2info = {}
concat_reviews = {}
for _review in cuisine_reviews:
    if not (_review['business_id'] in rest2info):
        rest2info[_review['business_id']] = {}
        concat_reviews[_review['business_id']] = ''
        for _, key in enumerate(rest[_review['business_id']]):
            rest2info[_review['business_id']][key] = rest[_review['business_id']][key]
        rest2info[_review['business_id']]['review_summary'] = ''
        rest2info[_review['business_id']]['reviews'] = []
    
    rest2info[_review['business_id']]['reviews'].append({'review_id': _review['review_id'],
                                               'user_id': _review['user_id'], 
                                               'stars': _review['stars'],
                                               'date': _review['date'],
                                               'text': _review['text']
                                              })
    concat_reviews[_review['business_id']] += _review['text']

for i, _id in enumerate(rest2info):
    rest2info[_id]['review_summary_gensim'] = summarize(concat_reviews[_id], word_count = 100)
    rest2info[_id]['review_summary_pytease'] = Summarize("chinese food and cuisine reviews", summarize(concat_reviews[_id]))
    # print(rest2info[_id]['review_summary_gensim'])
    # print('----------------------------------------------------')
    # print(rest2info[_id]['review_summary_pytease'])
    print('%d----------------------------------------------------'%(i))

# for _id in rest2info:
#     print(rest2info[_id]['review_summary_gensim'])
#     print('----------------------------------------------------')
#     print(rest2info[_id]['review_summary_pytease'])

with open(APP_FOLDER + 'restinfo.plk', 'wb') as f:
    pickle.dump(rest2info, f)