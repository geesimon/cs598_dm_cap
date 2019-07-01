import json
import pickle
import logging

path2files="../yelp_dataset_challenge_academic_dataset/"
path2buisness=path2files+"yelp_academic_dataset_business.json"
path2reviews=path2files+"yelp_academic_dataset_review.json"

def main():
    categories = set([])
    restaurant_ids = set([])
    cat2rid = {}
    rest2rate={}
    rest2revID = {}
    r = 'Restaurants'
    with open (path2buisness, 'r') as f:
        for line in f.readlines():
            business_json = json.loads(line)
            bjc = business_json['categories']
            #cities.add(business_json['city'])
            if r in bjc:
                if len(bjc) > 1:
                    restaurant_ids.add(business_json['business_id'])
                    categories = set(bjc).union(categories) - set([r])
                    stars = business_json['stars']
                    rest2rate[ business_json['business_id'] ] = stars
                    for cat in bjc:
                        if cat == r:
                            continue
                        if cat in cat2rid:
                            cat2rid[cat].append(business_json['business_id'])
                        else:
                            cat2rid[cat] = [business_json['business_id']]

    rest_review = {}
    with open (path2reviews, 'r') as f:
        for line in f.readlines():
            review_json = json.loads(line)
            review_json['text'] = review_json['text'].replace("\t", " ")\
                                                        .replace("\n", "")\
                                                        .replace("\r", "")\
                                                        .strip()
            
            if review_json['business_id'] in restaurant_ids:
                rest_review[review_json['review_id']] = review_json
                if review_json['business_id'] in rest2revID:
                    rest2revID[review_json['business_id']].append(review_json['review_id'])
                else:
                    rest2revID[review_json['business_id']] = [ review_json['review_id'] ]

    valid_cats = []
    for i, cat in enumerate(cat2rid):
        cat_total_reviews = 0
        for rid in cat2rid[cat]:
            #number of reviews for each of restaurants
            if rid in rest2revID:
                cat_total_reviews = cat_total_reviews + len(rest2revID[rid])

        if cat_total_reviews > 30:
            valid_cats.append(cat)

    for cat in valid_cats:
        cat_reviews = []
        for rest_id in cat2rid[cat]:
            if rest_id not in rest2revID:
                continue
            for review_id in rest2revID[rest_id]:
                cat_reviews.append(rest_review[review_id])
        with open ('categories/' + cat.replace('/', '-').replace(" ", "_") + ".pkl" , 'wb') as f:
            pickle.dump(cat_reviews)

if __name__=="__main__":
    main()
