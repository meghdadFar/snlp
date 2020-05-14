import json

from snlp.data_processing.processing import prepare_imdb_data


if __name__ == "__main__":

    with open("config/imdb_reviews.json") as json_file:
        config = json.load(json_file)

    print('Preparing data...')
    prepare_imdb_data(output_dir=config['imdb']['out_dir'], seq_length=500)
