from fasttext import train_supervised, load_model
from snlp.evaluation.metrics import evaluate_model
from snlp.filtering.eval_filtering import predict_dataframe
from matplotlib import pyplot as plt
import pandas as pd


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
    
    evaluation(path_to_testfile='./tmp-amazon/test.tsv',
               path_to_model='./tmp-amazon/classifier',
               path_to_cfx='./tmp-amazon/confusion_matrix.png')

    evaluation(path_to_testfile='./tmp-amazon/test_repl.tsv',
               path_to_model='./tmp-amazon/classifier_repl',
               path_to_cfx='./tmp-amazon/confusion_matrix_repl.png')