import os
import json
from typing import List
import pandas
from snlp.mwes.am import calculate_am
from snlp.mwes.mwe_utils import update_corpus, get_counts
from snlp import logger


class MWE(object):
    def __init__(self, df: pandas.DataFrame, text_column: str, mwe_types: List[str]=['NC'], output_dir: str='tmp') -> None:
        """Provide functionalities around MWEs, for unsupervised extraction of MWEs from text and replacing
        them in the corpus.

        Args:
            df: DataFrame with a text_column that contains the corpus.
            text_col: Specifies the column of DataFrame that contains the corpus.
            mwe_types: Types of MWEs. Can be any of [NC, JNC]
            output_dir: Output directory where counts, MWEs and corpus with replaced MWEs are stored.
            count_dir: Directory where count_file is sotred.
            count_file: File in which counts are sotred.
            mwe_dir: Directory where mwe_file is sotred.
            mwe_file: File in which MWEs are sotred.
        
        Returns:
            None
        """
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
    def extract_mwes(self, am: str='pmi') -> None:
        """
        Args:
            mwe_types: Types of MWEs. Can be any of [NC, JNC]
            am: The association measure to be used. Can be any of [pmi, npmi]

        Returns:
            None
        """
        with open(self.count_file, "r") as file:
            count_data = json.load(file)
        mwe_am_dict = calculate_am(count_data=count_data, am=am, mwe_types=self.mwe_types)
        try:
            os.mkdir(self.mwe_dir)
        except Exception as e:
            logger.error(e)
            raise e
        try:
            with open(self.mwe_file, "w") as file:
                json.dump(mwe_am_dict, file)
        except Exception as e:
            logger.error(e)
            raise e

    def replace_mwes(self, am_threshold=0.5, only_mwes=False, lower_case=False, output_path=None) -> None:
        """Replaces MWEs that are extracted by running extract_mwes in the corpus and stores the result
            in a new file that is specified by output_path.

            Args:
                am_threshold: MWEs with an am greater than or equal to this threshold are selected for replacement.
                only_mwes: Whether or not keep only MWEs and drop the rest of the text.
                lower_case: Whether or not lowercase the sentence before replacing MWEs. 
                output_path: Path to new corpus, where MWEs are replaced. 

            Returns:
                None
        """
        if not output_path:
            output_path = os.path.join(self.output_dir,'text_mwe_repl.csv')
        try:
            with open(self.mwe_file, "r") as file:
                mwe_type_mwe_am = json.load(file)
        except Exception as e:
            logger.error(e)
            logger.error(f"Make sure you have previously run extract_mwes() and {self.mwe_file} file was successfully created.")
            raise e
        new_df = update_corpus(mwe_type_mwe_am,
                            mwe_types=self.mwe_types,
                            df=self.df, 
                            am_threshold=am_threshold,
                            only_mwes=only_mwes,
                            lower_case=lower_case,
                            text_column=self.text_col)
        new_df.to_csv(output_path, sep='\t')

        
        
        



        