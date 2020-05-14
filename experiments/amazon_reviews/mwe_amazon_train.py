from fasttext import train_supervised, load_model
from snlp.evaluation.metrics import evaluate_model
from snlp.filtering.eval_filtering import predict_dataframe
from matplotlib import pyplot as plt
import pandas as pd

def train(path_to_trainfile, path_to_model, epochs):
    classification_model = train_supervised(path_to_trainfile, dim=100, epoch=epochs, wordNgrams=2, lr=0.01)
    classification_model.save_model(path_to_model)


if __name__ == "__main__":

    train(path_to_trainfile='./tmp-amazon/train.tsv', path_to_model='./tmp-amazon/classifier', epochs=200)
    train(path_to_trainfile='./tmp-amazon/train_repl.tsv', path_to_model='./tmp-amazon/classifier_repl', epochs=200)