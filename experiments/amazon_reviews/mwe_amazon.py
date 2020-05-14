import pandas as pd
import re

from snlp.mwes.am import get_counts, get_ams
from snlp.filtering.eval_filtering import clean_text

from nltk.tokenize import word_tokenize
from nltk.tokenize import ToktokTokenizer
toktok = ToktokTokenizer()

if __name__ == "__main__":

    def cleantext(text):
        tokens = text.split(" ")
        res = []
        for t in tokens:
            if re.match('^[a-zA-Z0-9!.,?\';:$/_-]+$', t):
                res.append(t)
        return ' '.join(res)

    # df = pd.read_csv('tmp-amazon/amazon_reviews.tsv', sep='\t')
    # df = df.dropna()
    
    # print('Tokenizing')
    # df.review_body = df.review_body.apply(word_tokenize).apply(lambda x : ' '.join(x))

    # print('Cleaning')
    # df.review_body = df.review_body.apply(cleantext)
    # df = df.replace('', float('nan'))
    # df = df.dropna()
    

    # print('Saving')
    # df.to_csv('tmp-amazon/amazon_reviews_tok_clean.tsv', sep='\t', index=False)

    # df = pd.read_csv('tmp-amazon/amazon_reviews_tok_clean.tsv', sep='\t')
    # print('Getting counts')
    # get_counts(df, text_column='review_body', output_dir='tmp-amazon')
    
    print('Calculating AMs')
    get_ams(path_to_counts='tmp-amazon')
