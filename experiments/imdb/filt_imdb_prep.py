import json
import torch

from snlp.data_processing.processing import dataset_to_corpus 
from torchtext import data
from nltk import word_tokenize
from torchtext import datasets


def prepare_imdb_data(output_dir, seq_length):
    """Loads the raw data train and test splits, lowers case, tokenizes and saves them as tsv. 

    """
    # Setting up fields
    TEXT = data.Field(lower=True, tokenize=word_tokenize, fix_length=seq_length)
    LABEL = data.LabelField(dtype=torch.float)

    # Splitting data
    train_ds, test_ds = datasets.IMDB.splits(TEXT, LABEL)
    print('Number of train examples: %d' %len(train_ds))
    print('Number of test examples: %d' %len(test_ds))

    dataset_to_corpus(train_ds, os.path.join(output_dir, 'imdb_train.tsv'))
    dataset_to_corpus(test_ds, os.path.join(output_dir, 'imdb_test.tsv'))


if __name__ == "__main__":

    with open("config/imdb_reviews.json") as json_file:
        config = json.load(json_file)

    print('Preparing data...')
    prepare_imdb_data(output_dir=config['imdb']['out_dir'], seq_length=500)
