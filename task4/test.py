import pickle
import re


def splitParagraphIntoSentences(paragraph):
    import re
    sentenceEnders = re.compile(r"""
        # Split sentences on whitespace between them.
        (?:               # Group for two positive lookbehinds.
          (?<=[.!?])      # Either an end of sentence punct,
        | (?<=[.!?]['"])  # or end of sentence punct and quote.
        )                 # End group of two positive lookbehinds.
        (?<!  Mr\.   )    # Don't end sentence on "Mr."
        (?<!  Mrs\.  )    # Don't end sentence on "Mrs."
        (?<!  Jr\.   )    # Don't end sentence on "Jr."
        (?<!  Dr\.   )    # Don't end sentence on "Dr."
        (?<!  Prof\. )    # Don't end sentence on "Prof."
        (?<!  Sr\.   )    # Don't end sentence on "Sr."
        \s+               # Split on whitespace between sentences.
        """, 
        re.IGNORECASE | re.VERBOSE)
    sentenceList = sentenceEnders.split(paragraph)
    return sentenceList

with open('./categories/Chinese.pkl', 'rb') as f:
    cuisine_reviews = pickle.load(f)

with open('ChineseDishes.txt', 'r') as f:
    dishes = {line.replace("\n", ""):{} for line in f.readlines()[0:100]}

for _dish in dishes:
    _dish2rest = set()
    _total_count = 0
    for _review in cuisine_reviews:
        _count = _review['text'].count(_dish)
        if _count > 0:
            s = splitParagraphIntoSentences(_review['text'])
            pattern = "\.?(?P<sentence>.*?" + _dish + ".*?)\."
            match = re.search(pattern, _review['text'])
            if match != None:
                print(match.group("sentence"))
            _total_count += _count
            if _review['business_id'] not in _dish2rest:
                _dish2rest.add(_review['business_id'])
    
    
    print("%s:%d:%d"%(_dish, _total_count, len(_dish2rest)))