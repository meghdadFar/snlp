
import os
import sys
import json

import pandas as pd
from fasttext import load_model

from snlp.data_processing.processing import predict_dataframe
from snlp.evaluation.metrics import evaluate_model


def evaluate_models(input_dir, model_dir):
        test_sets = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('test.tsv')]
        for ts in test_sets:
            z_value = None
            manual = None
            if '_filt_z' in ts:
                z_value = ts.split('_')[2]
            if '_manual_' in ts:
                manual = True

            print("z= %s" %z_value)
            print("Manual= %s" %manual)
            abs_path = os.path.join(input_dir, ts)
            
            if z_value and not manual:
                model_abs_path = os.path.join(model_dir, z_value+'_ft_classifier')
            elif not z_value and manual:
                model_abs_path = os.path.join(model_dir, 'manual_ft_classifier')
            elif not z_value and not manual:
                model_abs_path = os.path.join(model_dir, 'ft_classifier')
            else:
                sys.exit('Model cannot be based on both manual and automatic cleaning')

            ft_model = load_model(model_abs_path)

            df = pd.read_csv(abs_path, sep='\t', names=['label', 'text'])
            predictions, groundtruth = predict_dataframe(ft_model, df)
            res = evaluate_model(predictions, groundtruth)
            print(res)
            print('------------------------')


if __name__ == "__main__":

    with open("config/imdb_reviews.json") as json_file:
        config = json.load(json_file)


    print('Training models...')
    evaluate_models(config['imdb']['out_dir'], config['imdb']['out_dir'])