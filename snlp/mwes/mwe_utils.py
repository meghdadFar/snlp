import json
import collections
from typing import Dict, List
import pandas
import re
import tqdm
from snlp import logger


def replace_compunds(mwe_dict: Dict[str, Dict],
                    mwe_types: List[str],
                    df: pandas.DataFrame,
                    text_column: str,
                    am_threshold: float,
                    only_mwes: bool,
                    lower_case: bool) -> pandas.DataFrame:
    """Hyphenates the compounds in the corpus so that they are treated as a single token by downstream applications.

    Args:
        mwe_dict: Dictionary of MWE type for each type, unique MWEs to their count. E.g. {'NC': {'mwe1': 10}}.
        mwe_types: Types of MWEs to be replaced. Can be any of [NC, JNC].
        df: DataFrame comprising training data with a tokenized text column.
        text_column: Text (content) column of df.
        am_threshold: MWEs with an am greater than or equal to this threshold are selected for replacement.
        only_mwes: Whether or not keep only MWEs and drop the rest of the text.
        lower_case: Whether or not lowercase the sentence before replacing MWEs. 

    Returns:
        df (pandas.FataFrame)
    """
    good_mwes = set()
    for t in mwe_types:
        pmi_sorted_dict = mwe_dict[t]
        logger.info(f'Number of all MWEs of type {t}: {len(pmi_sorted_dict)}')
        for k,v in pmi_sorted_dict:
            if v >= am_threshold:
                good_mwes.add(k)
            else:
                break
        logger.info('Number of MWEs to be replaced in corpus based on the association threshold: %d' %len(good_mwes))

    logger.info('Replacing compounds in text')
    new_text = []
    for sent in tqdm.tqdm(df[text_column]):
        sent = (sent.lower() if lower_case else sent)
        bigrams = get_ngrams(sent, 2)
        if only_mwes:
            tmp = ''
            for bg in bigrams:
                if bg in good_mwes:
                    tmp += (bg.split(' ')[0]+'-'+bg.split(' ')[1]+' ')
            if tmp != '':
                sent = tmp.strip()
        else:
            for bg in bigrams:
                if bg in good_mwes:
                    sent = re.sub(bg, bg.split(' ')[0]+'-'+bg.split(' ')[1], sent)
        new_text.append(sent)
    df[text_column] = new_text
    return df


def get_ngrams(sentence: str, n: int) -> List:
    """Extracts n-grams from sentence.
    
    Args:
        sentence: Input sentence from which n-grams are to be extracted.
        n: Size of n-grams.

    Returns:
        ngrams: List of extracted n-grams.
    """
    ngrams = []
    try:
        tokens = sentence.split(" ")
    except Exception as E:
        logger.error(E)
        logger.error(f'Input sentence {sentence} cannot be spilitted around space. No n-gram is extracted.')
        return ngrams
    for i in range(len(tokens)-n+1):
        ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams