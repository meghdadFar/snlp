import os
from fasttext import train_supervised


def train_models(input_dir, output_dir):
        train_sets = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('train.tsv')]
        for ts in train_sets:
            z_value = None
            manual = None
            if '_filt_z' in ts:
                z_value = ts.split('_')[2]
            if '_manual_' in ts:
                manual = True

            abs_path = os.path.join(input_dir, ts)
            classification_model = train_supervised(abs_path,
                                                    dim=100,
                                                    epoch=500,
                                                    wordNgrams=2,
                                                    lr=0.01)
            
            if z_value and not manual:
                classification_model.save_model(os.path.join(output_dir, z_value+'_ft_classifier'))
            elif not z_value and manual:
                classification_model.save_model(os.path.join(output_dir, 'manual_ft_classifier'))
            elif not z_value and not manual:
                classification_model.save_model(os.path.join(output_dir, 'ft_classifier'))
            else:
                sys.exit('Filterset cannot be both manual and automatic')



if __name__ == "__main__":


    raw_dir = 'tmp'
    output_dir = 'tmp3'
    prefix = 'imdb'
    zs = [3, 4, 5]
    regex_pattern = '^[a-zA-Z0-9!.?/]+$'
    method = 'manual'

    print('Training models...')
    train_models(output_dir, output_dir)