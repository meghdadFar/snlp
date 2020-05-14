import pandas as pd
from snlp.mwes.update_corpus import replace_compunds

orig_df = pd.read_csv('./tmp-amazon/amazon_reviews_tok_clean.tsv', sep='\t')
orig_df = orig_df.replace('', float('nan'))
orig_df = orig_df.dropna()

new_df = replace_compunds(path_to_compounds='tmp-amazon/nn_pmi.json',
                          df=orig_df,
                          text_column='review_body',
                          am_threshold=0.3,
                          only_compounds=False, 
                          lower_case=True)

new_df.to_csv('./tmp-amazon/amazon_review_tok_clean_repl.tsv', index=False, sep='\t')