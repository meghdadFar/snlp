import json
import re

import pandas as pd

from sklearn import preprocessing
from nltk.tokenize import word_tokenize
from nltk.tokenize import ToktokTokenizer

toktok = ToktokTokenizer()


with open("config/amazon_reviews.json") as json_file:
        config = json.load(json_file)

amazon_reviews = pd.read_csv(config["amazon"]["raw_file"], sep='\t', error_bad_lines=False)

# Adjust and interpret labels 
hlpful_vote_count = amazon_reviews.helpful_votes.tolist()
hlpful_vote_std = preprocessing.scale(hlpful_vote_count)
hlpful_vote_std_round = [round(x, 2) for x in hlpful_vote_std]
amazon_reviews['helpful_votes_std'] = hlpful_vote_std_round
amazon_reviews['reliability'] = amazon_reviews.helpful_votes.apply(lambda x :'low' if x <= 5 else ('moderate' if x <= 20 else ('substantial' if x <= 100 else 'high')))
amazon_reviews['sentiment'] = amazon_reviews.star_rating.apply(lambda x : 'neg' if x <= 2 else ('neutral' if x == 3 else 'pos'))


# Tokenize and clean review_body
amazon_reviews.review_body = amazon_reviews.review_body.apply(word_tokenize).apply(lambda x : ' '.join(x))
def cleantext(text):
    tokens = text.split(" ")
    res = []
    for t in tokens:
        if re.match('^[a-zA-Z0-9!.,?\';:$/_-]+$', t):
            res.append(t)
    return ' '.join(res)
amazon_reviews.review_body = amazon_reviews.review_body.apply(cleantext)

# Remove rows with empty field
amazon_reviews = amazon_reviews.replace('', float('nan'))
amazon_reviews = amazon_reviews.dropna()

# Save results
amazon_reviews[['review_body', 'star_rating', 
                'sentiment', 'helpful_votes', 
                'helpful_votes_std','reliability']].to_csv(config["amazon"]["preped_file"], sep='\t', index=False)



                
                