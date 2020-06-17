import json
import pandas as pd

from snlp.data_analysis import text_analysis


with open("config/amazon_reviews.json") as json_file:
    config = json.load(json_file)

amazon_reviews = pd.read_csv(config['amazon']['preped_file'], sep='\t', error_bad_lines=False)
amazon_reviews = amazon_reviews.sample(100)

text_analysis.generate_report(df=amazon_reviews, 
                              out_dir=config['amazon']['analysis_dir'],
                              text_col='review_body',
                              label_cols=[('sentiment', 'categorical'), ('reliability', 'categorical'), ('star_rating', 'numerical')],
                              pos_tags=['NN', 'JJ', 'VB'])

# label_cols=[('sentiment', 'categorical'), ('reliability', 'categorical'), ('star_rating', 'numerical')]

