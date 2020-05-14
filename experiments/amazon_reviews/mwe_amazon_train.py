from fasttext import train_supervised, load_model
from snlp.evaluation.metrics import evaluate_model
from snlp.filtering.eval_filtering import predict_dataframe
from matplotlib import pyplot as plt
import pandas as pd
import json
import os


def train(path_to_trainfile, path_to_model, epochs):
    classification_model = train_supervised(path_to_trainfile, dim=100, epoch=epochs, wordNgrams=2, lr=0.01)
    classification_model.save_model(path_to_model)


if __name__ == "__main__":

    with open("config/amazon_reviews.json") as json_file:
        config = json.load(json_file)

    train(path_to_trainfile=os.path.join(config['amazon']['test_test_dir'], 'train.tsv'), 
          path_to_model=os.path.join(config['amazon']['test_test_dir'], 'classifier'), 
          epochs=200)

    train(path_to_trainfile=os.path.join(config['amazon']['test_test_dir'], 'train_repl.tsv'), 
          path_to_model=os.path.join(config['amazon']['test_test_dir'], 'classifier_repl'), 
          epochs=200)