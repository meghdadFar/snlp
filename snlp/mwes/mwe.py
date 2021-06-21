from logging import log
import os
import json
from typing import List
import pandas
from snlp.mwes.am import calculate_am
from snlp.mwes.counter import get_counts
from snlp import logger

class MWE(object):
    """ Provide functionalities for an unsupervised extraction of noun compound MWEs from text.
    1. Functions extract MWEs from training data via PMI, NPMI, and SDMA from training corpus with application of POS tags keeping meaningful sequences (options: keep only NNNNs, JJNN, 
    or keeping all grammatically meaningful)
    2. Function to hyphenate the above MWEs in the corpus before creation of the embeddings so that MWEs get an embedding.
    3. If 2 is not applied, function to create summed embeddings for MWEs extracted in 1, and include them in fasttext/or other we model. 
    4. Function to find above MWEs (extracted in 1) in test data and hyphenate them. 
    """
    
    def __init__(self, df: pandas.DataFrame, text_column: str, mwe_types: List[str], output_dir: str='tmp') -> None:
        self.df = df
        self.text_col = text_column
        self.mwe_types = mwe_types
        self.output_dir = output_dir
        self.count_dir = os.path.join(self.output_dir, 'counts')
        self.count_file = os.path.join(self.count_dir, 'count_data.json')
        self.mwe_dir = os.path.join(self.output_dir, 'mwes')
        self.mwe_file = os.path.join(self.count_dir, 'mwe_data.json')

    def build_counts(self) -> None:
        """Create various count files to be used by downstream methods 
        by calling snlp.mwes.counter.get_counts.

        Args:
            None
        Returns:
            None
        """
        logger.info('Creating counts...')
        res = get_counts(df=self.df, text_column=self.text_col, mwe_types=self.mwe_types)
        try:
            os.mkdir(self.count_dir)
        except Exception as e:
            logger.error(e)
            raise e
        try:
            with open(self.count_file, "w") as file:
                json.dump(res, file)
        except Exception as e:
            logger.error(e)
            raise e


    def extract_mwes(self, mwe_types: str=['NC'], am: str='pmi') -> None:
        """
        Args:
            mwe_types: Types of MWEs. Can be any of [NC, JNC]
            am: 

        Returns:
            None
        """
        with open(self.count_file, "w") as file:
            count_data = json.load(file)

        am_data = calculate_am(count_data=count_data, am=am, mwe_types=mwe_types)
        try:
            os.mkdir(self.mwe_dir)
        except Exception as e:
            logger.error(e)
            raise e
        try:
            with open(self.mwe_file, "w") as file:
                json.dump(am_data, file)
        except Exception as e:
            logger.error(e)
            raise e