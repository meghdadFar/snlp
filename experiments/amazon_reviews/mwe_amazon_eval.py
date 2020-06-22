import pandas as pd
import json
import os

from fasttext import train_supervised, load_model
from snlp.evaluation.metrics import evaluate_model
from snlp.evaluation.functions import predict_dataframe
from matplotlib import pyplot as plt


def evaluation(path_to_testfile, path_to_model, path_to_cfx):
    df = pd.read_csv(path_to_testfile, sep='\t', names=['label', 'text'])
    ft_model = load_model(path_to_model)

    predictions, groundtruth = predict_dataframe(ft_model, df)
    res = evaluate_model(predictions, groundtruth)
    res = evaluate_model(predictions, groundtruth, path_to_cfx)
    print(res.micro_p)
    print(res.macro_p)
    print(res.accuracy)


if __name__ == "__main__":

    with open("config/amazon_reviews.json") as json_file:
        config = json.load(json_file)
    
    evaluation(path_to_testfile=os.path.join(config['amazon']['test_test_dir'], 'test.tsv'),
               path_to_model=os.path.join(config['amazon']['test_test_dir'], 'classifier'),
               path_to_cfx=os.path.join(config['amazon']['test_test_dir'], 'confusion_matrix.png'))

    evaluation(path_to_testfile=os.path.join(config['amazon']['test_test_dir'], 'test_repl.tsv'),
               path_to_model=os.path.join(config['amazon']['test_test_dir'], 'classifier_repl'),
               path_to_cfx=os.path.join(config['amazon']['test_test_dir'], 'confusion_matrix_repl.png'))