from sklearn.model_selection import train_test_split 
import pandas as pd
import json
import os


def prepare_train_test(path_to_df, path_to_train, path_to_test, label_col='reliability'):
    df = pd.read_csv(path_to_df, sep='\t')
    df = df.replace('', float('nan'))
    df = df.dropna()
    df.review_body = df.review_body.apply(lambda x : x.lower())
    # Adjust reliability score
    df['reliability'] = df.helpful_votes.apply(lambda x :'low' if x == 0 else 
                                            ('moderate' if x <= 5 else 
                                            ('substantial' if x <= 20 else 
                                            ('high' if x <= 100 else 'very_high')
                                            )))
    df[label_col] = df[label_col].apply(lambda x : "__label__"+x)
    train_df, test_df = train_test_split(df, test_size=0.3, stratify=df[label_col], random_state=555)
    # 'review_body	star_rating	sentiment	helpful_votes	helpful_votes_std	reliability'
    train_df[[label_col, 'review_body']].to_csv(path_to_train, header=False, index=False, sep='\t')
    test_df[[label_col, 'review_body']].to_csv(path_to_test, header=False, index=False, sep='\t')


if __name__ == "__main__":

    with open("config/amazon_reviews.json") as json_file:
        config = json.load(json_file)
    
    
    prepare_train_test(path_to_df=config['amazon']['preped_file'], 
                       path_to_train=os.path.join(config['amazon']['test_test_dir'], 'train.tsv'), 
                       path_to_test=os.path.join(config['amazon']['test_test_dir'], 'test.tsv'),
                       label_col='sentiment')

    prepare_train_test(path_to_df=config['amazon']['mwe_rep_file'], 
                       path_to_train=os.path.join(config['amazon']['test_test_dir'], 'train_repl.tsv'), 
                       path_to_test=os.path.join(config['amazon']['test_test_dir'], 'test_repl.tsv'),
                       label_col='sentiment')

