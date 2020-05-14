import pandas as pd
import json

from snlp.mwes.am import get_counts, get_ams


with open("config/amazon_reviews.json") as json_file:
        config = json.load(json_file)

get_ams(path_to_counts=config["amazon"]["count_dir"])