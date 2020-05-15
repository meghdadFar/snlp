import pandas as pd
import json

from snlp.mwes.am import get_counts, get_ams


with open("config/amazon_reviews.json") as json_file:
        config = json.load(json_file)

df = pd.read_csv(config["amazon"]["preped_file"], sep='\t')
get_counts(df, text_column=config["amazon"]["review_body"], output_dir=config["amazon"]["count_dir"])
