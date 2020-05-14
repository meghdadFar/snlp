





if __name__ == "__main__":
    
    raw_dir = 'tmp'
    output_dir = 'tmp3'
    prefix = 'imdb'
    zs = [3, 4, 5]
    regex_pattern = '^[a-zA-Z0-9!.?/]+$'
    method = 'manual'

    print('Preparing data...')
    prepare_raw_data(input_dir=raw_dir, prefix=prefix, seq_length=500)

    
    if CLEANING:
        if method == 'automatic':
            print('Creating zscore-based filtersets...')
            z_map = create_filterset_map(os.path.join(raw_dir, prefix+'_train.tsv'), 
                                output_dir=output_dir, 
                                zs=zs)
            print('Cleaning sets...')
            run_cleaning_map(raw_dir, output_dir, z_map, regex_pattern)

        elif method == 'manual':

            print('Creating manual filtersets...')
            filter_set = create_filterset(os.path.join(raw_dir, prefix+'_train.tsv'), 
                                output_dir=output_dir)
            print('Cleaning sets...')
            run_cleaning_set(raw_dir, output_dir, filter_set, regex_pattern)


    if TRAINING:
        print('Training models...')
        train_models(output_dir, output_dir)
    

    if TESTING:
        evaluate_models(output_dir, output_dir)