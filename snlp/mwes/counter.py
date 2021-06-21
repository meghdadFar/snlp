import pandas
import nltk
import tqdm
import re
from collections import Counter
from typing import List, Tuple


def get_counts(df: pandas.DataFrame, text_column: str, mwe_types: List[str], output_dir: str) -> dict:
    """Read a corpus in pandas.DataFrame format and generates all counts necessary for calculating AMs.

    Args:
        df (pandas.DataFrame): DataFrame with input data, which contains a column with text content
                               from which compounds and their counts are extracted.
        text_column: Name of the column the contains the text content.
        mwe_types: Types of MWEs. Can be any of [NC, JNC]
        output_dir: Path to output dir where calculated counts will be stored.

    Returns:
        res: Dictionary of mwe_types to dictionary of individual mwe within that type and their count.
            E.g. {'NC':{'climate change': 10, 'brain drain': 3}, 'JNC': {'black sheep': 3, 'red flag': 2}}
    """
    res = {}
    for mt in mwe_types:
        res[mt]={}
    res['WORDS']={}

    for sent in tqdm.tqdm(df[text_column]):

        tokens = sent.split(" ")
        word_count_dict = Counter(tokens)
        
        for k, v in word_count_dict.items():
            if k in res['WORDS']:
                res['WORDS'][k] += v
            else:
                res['WORDS'][k] = v
        
        for mt in mwe_types:
            count = 0
            mwes_count_dic = extract_ncs_from_sent(sent, mwe_types=mt)
            for k, v in mwes_count_dic.items():
                if k in res[mt]:
                    res[mt][k] += v
                else:
                    res[mt][k] = v
        return res


def extract_ncs_from_sent(tokens: List[str], type: str) -> Tuple(dict, dict, dict):
    """Extract two-word noun compounds from tokenized input.

    Args:
        tokens: A tokenized sentence.
        type: Type of MWE. Any of [NC, JNC].

    Returns:
        mwes_count_dic: Dictionary of compounds to their count.
    """
    if not isinstance(tokens, list):
        raise TypeError("Input argument \"tokens\" must be a list of string.")

    if len(tokens) == 0:
        return
    
    mwes = []

    postag_tokens = nltk.pos_tag(tokens)
    
    w1_pos_tags = []
    w2_pos_tags = []

    if type == 'NC':
        w1_pos_tags = ["NN", "NNS"]
        w2_pos_tags = ["NN", "NNS"]
    elif type == 'JNC':
        w1_pos_tags = ["JJ"]
        w2_pos_tags = ["NN", "NNS"]

    for i in range(len(postag_tokens) - 1):
        w1 = postag_tokens[i]
        if w1[1] not in w1_pos_tags:
            continue
        else:
            w2 = postag_tokens[i + 1]
            if not re.match("[a-zA-Z0-9]{2,}", w1[0]) or not re.match("[a-zA-Z0-9]{2,}", w2[0]):
                continue

            if w2[1] in w2_pos_tags:
                if i + 2 < len(postag_tokens):
                    w3 = postag_tokens[i + 2]
                    if w3 not in ["NN", "NNS"]: 
                        mwes.append(w1[0] + " " + w2[0])
                else:
                    mwes.append(w1[0] + " " + w2[0])
                    
    mwes_count_dic = Counter(mwes)    
    return mwes_count_dic