import pandas as pd
import json
import os

from snlp.mwes.update_corpus import replace_compunds


with open("config/amazon_reviews.json") as json_file:
        config = json.load(json_file)

orig_df = pd.read_csv(config["amazon"]["preped_file"], sep='\t')
orig_df = orig_df.replace('', float('nan'))
orig_df = orig_df.dropna()

new_df = replace_compunds(path_to_compounds=os.path.join(config["amazon"]["count_dir"],'nn_pmi.json'),
                          df=orig_df,
                          text_column='review_body',
                          am_threshold=0.3,
                          only_compounds=False, 
                          lower_case=True)

new_df.to_csv(config["amazon"]["mwe_rep_file"], index=False, sep='\t')