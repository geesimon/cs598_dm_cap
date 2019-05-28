import math
import json
import pickle
import random
from gensim import models
from gensim import matutils
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from nltk.tokenize import sent_tokenize
import glob
import argparse
import os
path2files="../yelp_dataset_challenge_academic_dataset/"
path2buisness=path2files+"yelp_academic_dataset_business.json"
path2reviews=path2files+"yelp_academic_dataset_review.json"
path2reviewdump = "reviews/reviews.dat"


def main(save_sample, save_categories):
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
                    #print(bjc)
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

    print("saving restaurant ratings")
    with open ( 'restaurantIds2ratings.txt', 'w') as f:
        for key in rest2rate:
            f.write( key + " " + str(rest2rate[key]) + "\n")
    #clearing from memory
    rest2rate=None
    with open('data_cat2rid.pickle', 'wb') as f:
        pickle.dump(cat2rid,f)

    rest_review = []
    with open (path2reviews, 'r') as f:
        for line in f.readlines():
            review_json = json.loads(line)
            if review_json['business_id'] in restaurant_ids:
                rest_review.append(review_json['text'].replace("\n", "").replace("\r", "").strip())
                if review_json['business_id'] in rest2revID:
                    rest2revID[ review_json['business_id'] ].append(review_json['review_id'])
                else:
                    rest2revID[ review_json['business_id'] ] = [ review_json['review_id'] ]

    with open('data_rest2revID.pickle', 'wb') as f:
        pickle.dump(rest2revID,f)

    #Save the First 100000 resturant reviews
    sample_size = 100000
    random.seed(19720816)
    sample_text = [rest_review[i] for i in sorted(random.sample(range(len(rest_review)), sample_size))]
    
    with open ("rest_review_sample_100000.txt", 'wb') as f:
        f.write('\n'.join(sample_text).encode('ascii','ignore'))

    nz_count = 0
    valid_cats = []
    for i, cat in enumerate(cat2rid):
        cat_total_reviews = 0
        for rid in cat2rid[cat]:
            #number of reviews for each of restaurants
            if rid in rest2revID:
                cat_total_reviews = cat_total_reviews + len(rest2revID[rid])

        if cat_total_reviews > 30:
            nz_count = nz_count + 1
            valid_cats.append(cat)
            #print( cat, cat_total_reviews)

    #print nz_count, ' non-zero number of reviews in categories out of', len(cat2rid), 'categories')
    #x = range(nz_count)
    print("sampling categories")
    sample_rid2cat={}
    for cat in valid_cats:
        for rid in cat2rid[cat]:
            if rid in rest2revID:
                if rid not in sample_rid2cat:
                    sample_rid2cat[rid] = []
                sample_rid2cat[rid].append(cat)
    #remove from memory
    rest2revID=None
    #    print (len(sample_rid2cat), len(cat2rid), len(valid_cats), len(cat_sample))
    
    print("reading from reviews file...")
    #ensure categories is a directory
    sample_cat2reviews={}
    sample_cat2ratings={}
    num_reviews = 0
    all_rest_reviews = []
    with open (path2reviews, 'r') as f:
        for line in f.readlines():
            review_json = json.loads(line)
            rid = review_json['business_id']
            if rid in sample_rid2cat:
                num_reviews += 1
                _stars = review_json['stars']
                _review = review_json['text'].replace("\n", "").replace("\r", "").strip()
                all_rest_reviews.append(_review)
                for rcat in sample_rid2cat [ rid ]:
                    if rcat in sample_cat2reviews:
                        sample_cat2reviews [ rcat ].append(_review)
                        sample_cat2ratings [ rcat ].append( str(_stars) )
                    else:
                        sample_cat2reviews [ rcat ] = [_review]
                        sample_cat2ratings [ rcat ] = [ str(_stars) ]
                
        if save_categories:
            print("saving categories")
            #save categories
            for cat in sample_cat2reviews:
                with open ('categories/' + cat.replace('/', '-').replace(" ", "_") + ".txt" , 'wb') as f:
                    f.write('\n'.join(sample_cat2reviews[cat]).encode('ascii','ignore'))

    if save_sample:  
        print("sampling restaurant reviews")
        with open(path2reviewdump, 'wb') as f:
            f.write('\n'.join(all_rest_reviews).encode('ascii','ignore'))
        #save sample for restaurant reviews
        sample_size = min(100000, num_reviews)
        rev_sample = random.sample(range(num_reviews), sample_size)
        my_sample_v2 = []
        sample_ratings = []
        sorted_rev_sample = sorted(rev_sample)
        count = 0
        max_bound = 0
        for cat in sample_cat2reviews:
            print(cat)
            new_max_bound = max_bound + len(sample_cat2reviews[cat])
            while count < sample_size and sorted_rev_sample[count] < new_max_bound:
                my_sample_v2.append( sample_cat2reviews[cat][ sorted_rev_sample[count] - max_bound ].replace("\n", " ").strip() )
                sample_ratings.append( sample_cat2ratings[cat][ sorted_rev_sample[count] - max_bound ] )
                count = count + 1
            max_bound = new_max_bound
            #if count in rev_sample:
            #    my_sample.append(rev.replace("\n", " ").strip())
            #count = count + 1

        # with open ("review_sample_100000.txt", 'wb') as f:
        #     f.write('\n'.join(my_sample_v2).encode('ascii','ignore'))
            
        # with open ("review_ratings_100000.txt", 'w') as f:
        #     f.write('\n'.join(sample_ratings))
            

def sim_matrix():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    K_clusters = 10
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                     min_df=2, stop_words='english',
                                     use_idf=True)

    
    
    if not os.path.isdir("categories"):
        print("you need to generate the cuisines files 'categories' folder first")
        return
    
    text = []
    c_names = []
    # cat_list = glob.glob ("categories/*")
    cat_list = glob.glob("cuisines/*")
    cat_size = len(cat_list)
    if cat_size < 1:
        print("you need to generate the cuisines files 'categories' folder first")
        return
    
    sample_size = min(50, cat_size)
    cat_sample = sorted( random.sample(range(cat_size), sample_size) )
    #print (cat_sample)
    count = 0
    for i, item in enumerate(cat_list):
        if i == cat_sample[count]:
            li =  item.split('/')
            cuisine_name = li[-1]
            c_names.append(cuisine_name[:-4].replace("_"," "))
            with open ( item ) as f:
                text.append(f.read().replace("\n", "").replace("\r",""))
            count = count + 1
        
        if count >= len(cat_sample):
            print("generating cuisine matrix with:%d cuisines"%(count))
            break

    if len(text) < 1:
        print("the 'categories' folder does not contain any cuisines. Run this program ussing the '--cuisine' option")
    t0 = time()
    print("Extracting features from the training dataset using a sparse vectorizer")
    X = vectorizer.fit_transform(text)
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)

    corpus = matutils.Sparse2Corpus(X,  documents_columns=False)
    lda = models.ldamodel.LdaModel(corpus, num_topics=100)

    doc_topics = lda.get_document_topics(corpus)
    cuisine_matrix = [] #similarity of topics
    # computing cosine similarity matrix
    for i, doc_a in enumerate(doc_topics):
        #print (i)
        sim_vecs = []
        for j , doc_b in enumerate(doc_topics):
            w_sum = 0
            if ( i <= j ):
                norm_a = 0
                norm_b = 0
                
                for (my_topic_b, weight_b) in doc_b:
                    norm_b = norm_b + weight_b*weight_b

                for (my_topic_a, weight_a) in doc_a:
                    norm_a = norm_a + weight_a*weight_a
                    for (my_topic_b, weight_b) in doc_b:
                        if ( my_topic_a == my_topic_b ):
                            w_sum = w_sum + weight_a*weight_b

                norm_a = math.sqrt(norm_a)
                norm_b = math.sqrt(norm_b)
                denom = (float) (norm_a * norm_b)
                if denom < 0.0001:
                    w_sum = 0
                else:
                    w_sum = w_sum/(denom)
            else:
                w_sum = cuisine_matrix[j][i]
            sim_vecs.append(w_sum)

        cuisine_matrix.append(sim_vecs)

    with open( 'cuisine_sim_matrix.csv', 'w') as f:
        for i_list in cuisine_matrix:
            s = ""
            my_max = max(i_list)
            for tt in i_list:
                s = s+str(tt/my_max) + " "
            s = s.strip()
            f.write(",".join(s.split())+"\n") #should the list be converted to m

    
    with open('cuisine_indices.txt', 'w') as f:
        f.write( "\n".join(c_names))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='This program transforms the Yelp data and saves the cuisines in the category directory. It also samples reivews from Yelp. It can also generates a cuisine similarity matrix.')
    
    parser.add_argument('--cuisine', action='store_true',
                       help='Saves a sample (10) of the cuisines to the "categories" directory. For Task 2 and 3 you will experiment with individual cuisines. This option allows you to generate a folder that contains all of the cuisines in the Yelp dataset. You can run this multiple times to generate more samples or if your machine permits you can change a sample parameter in the code.')
    parser.add_argument('--sample', action='store_true',
                       help='Sample a subset of reviews from the yelp dataset which could be useful for Task 1. This will samples upto 100,000 restaurant reviews from 10 cuisines and saves the output in "review_sample_100000.txt", it also saves their corresponding raitings in the "review_ratings_100000.txt" file. You can run this multiple times to get several different samples.')
    parser.add_argument('--matrix', action='store_true',
                       help='Generates the cuisine similarity matrix which is used for Task 2. First we apply topic modeling to a sample (30) of the cuisines in the "categories" folder and measures the cosine similarity of two cuisines from their topic weights. This might take from half-an-hour to several hours time depending on your machine. The number of topics is 20 and the default number of features is 10000.')
    parser.add_argument('--all', action='store_true',
                       help='Does all of the above.')

    
    args = parser.parse_args()        
    if args.all or (args.sample and args.cuisine):
        print("saving sample and cuisine")
        main(True,True)
    elif args.sample:
        print("generating sample")
        main(args.sample, args.cuisine)
    elif args.cuisine:
        print("generating cuisine")
        main(args.sample, args.cuisine)

    if args.matrix or args.all:
        print("generating cuisine matrix")
        sim_matrix()
    #main()
