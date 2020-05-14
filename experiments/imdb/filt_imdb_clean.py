import os
import traceback

from tqdm import tqdm
from snlp.data_processing.processing import clean_dataset, create_filterset_map, create_filterset




def run_cleaning_map(raw_home, output_home, z_map, regex_pattern):
    """Run cleaning over a set of zs and for train and test sets independently. 

    Args:
        raw_home (str): Path from which the original training and test sets can be read. 
        output_home (str): Path in which cleaned (and filtered) training and test sets are written. 
        path_to_raw_df (str): Path to the dataframe from which the filtersets are supposed to be extracted. 
        z_map (dict): Dict of z score and its corresponding filterset. Each will lead to new fiterset, train and test files.
    
    Returns:
    bool: The return value. True for success, False otherwise.
    
    """

    try:
        cv_sets = [f for f in os.listdir(raw_home) if os.path.isfile(os.path.join(raw_home, f)) and f.endswith('.tsv')]
        
        for set_name in tqdm(cv_sets):
            clean_dataset(os.path.join(raw_home, set_name),
                        os.path.join(output_home, 'clean_'+set_name),
                        regex_pattern=regex_pattern)
            for z, filt_words in z_map.items():
                clean_dataset(os.path.join(raw_home, set_name),
                            os.path.join(output_home, 'clean_filt_z'+str(z)+'_'+set_name),
                            regex_pattern=regex_pattern, filter_set=filt_words)
        return True
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False



def run_cleaning_set(raw_home, output_home, filterset, regex_pattern):
    """Run cleaning for train and test sets independently. 
    
    Args:
        raw_home (str): Path from which the original training and test sets can be read. 
        output_home (str): Path in which cleaned (and filtered) training and test sets are written. 
        path_to_raw_df (str): Path to the dataframe from which the filtersets are supposed to be extracted. 
        zs (list): Set of filter words. Used to clean cross validation splits. 
    
    Returns:
    bool: The return value. True for success, False otherwise.
    
    """

    try:
        cv_sets = [f for f in os.listdir(raw_home) if os.path.isfile(os.path.join(raw_home, f)) and f.endswith('.tsv')]
        
        for set_name in tqdm(cv_sets):
            clean_dataset(os.path.join(raw_home, set_name),
                        os.path.join(output_home, 'clean_'+set_name),
                        regex_pattern=regex_pattern)
            
            clean_dataset(os.path.join(raw_home, set_name),
                        os.path.join(output_home, 'clean_filt_manual_'+set_name),
                        regex_pattern=regex_pattern, filter_set=filterset)
        return True
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False


if __name__ == "__main__":


    raw_dir = 'tmp'
    output_dir = 'tmp3'
    prefix = 'imdb'
    zs = [3, 4, 5]
    regex_pattern = '^[a-zA-Z0-9!.?/]+$'
    method = 'manual'

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