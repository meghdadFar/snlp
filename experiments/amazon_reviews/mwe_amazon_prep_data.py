import pandas as pd
from sklearn import preprocessing

# Amazon Product Review 
amazon_reviews = pd.read_csv('/Users/meghdad/Code/data/amazon_reviews_us_Camera_v1_00.tsv', sep='\t', error_bad_lines=False)
hlpful_vote_count = amazon_reviews.helpful_votes.tolist()
hlpful_vote_std = preprocessing.scale(hlpful_vote_count)
hlpful_vote_std_round = [round(x, 2) for x in hlpful_vote_std]
amazon_reviews['helpful_votes_std'] = hlpful_vote_std_round

amazon_reviews['reliability'] = amazon_reviews.helpful_votes.apply(lambda x :'low' if x <= 5 else ('moderate' if x <= 20 else ('substantial' if x <= 100 else 'high')))
amazon_reviews['sentiment'] = amazon_reviews.star_rating.apply(lambda x : 'neg' if x <= 2 else ('neutral' if x == 3 else 'pos'))

amazon_reviews[['review_body', 'star_rating', 
                'sentiment', 'helpful_votes', 
                'helpful_votes_std','reliability']].to_csv('tmp-amazon/amazon_reviews.tsv', sep='\t', index=False)