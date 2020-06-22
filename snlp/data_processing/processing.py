import torch
import re
import nltk
import pandas as pd
import os
import sys
import traceback


from snlp.filtering.filters import WordFilter
from snlp.evaluation.metrics import evaluate_model
from nltk import word_tokenize
from fasttext import train_supervised, load_model

from tqdm import tqdm
from os import listdir


def preprocess_dataframe(path_to_input, path_to_output, regex_pattern, drop, replace, filter_set=None):
    """Remove unwanted patterns (and filter insignificant words). 
    """
    df = pd.read_csv(path_to_input, sep='\t', names=['label', 'text'])
    df.text = df.text.apply(lambda x: x.split(' '))
    df.text = df.text.apply(clean_text, args=(regex_pattern, drop, replace,))

    if filter_set:
        df.text = df.text.apply(lambda x: x.split(' ')) # TODO Also done above; Make more efficient. 
        df.text = df.text.apply(filter_text, args=(filter_set,))
    df.to_csv(path_to_output, index=False, header=False, sep='\t')
    

def clean_text(text, regex_pattern, drop, replace, maxlen=15):
    """ Tokenizes and cleans text, by matching it against regex_pattern and droping and replacing provided patterns. 
    
    Parameters
        text (string): Input text.
        regex_pattern (string): Allowed patterns e.g. [a-zA-Z]
        drop (set): Set of patterns that should be dropeed from text.
        replace (dict): Dictionary of pattern --> replacement.
    
    Returns:
        out_text (string): Tokenized and cleaned up text with respect to all above criteria. 

    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    if len(text) == 0:
        raise ValueError("Input must be a non empty string.")
    
    # Drop unwanted tokens: Replace them with space, then replace resulting \s{2, } with one space
    for d in drop: # d = e.g. <br/>
        if re.search(d, text):
            text = re.sub(d, ' ', text)
    text = re.sub('\s{2,}', ' ', text)

    for k,v in replace.items():
        if re.search(k, text):
            text = re.sub(k , v, text)

    tokens = word_tokenize(text)
    out_tokens = []
    for t in tokens:
        if len(t) < maxlen:
            if re.match(regex_pattern, t):
                out_tokens.append(t)
    out_text = ' '.join(out_tokens)
    return out_text


def filter_text(text, filter_set):
    """
    Parameters
    text (list): tokenized text
    """
    if not isinstance(text, list):
        raise TypeError("Input must be a list.")
    if len(text) == 0:
        raise ValueError("Input must be a non empty list.")

    res = ' '.join([t for t in text if t not in filter_set])
    return res


def save_filterset_tofile(filter_set, path):
    with open(path, "w") as f:
        for word in filter_set:
            f.write(word + '\n')
    f.close()


def create_filterset_map(p2_raw_train_df, output_dir, zs=[3]):
        original_df = pd.read_csv(p2_raw_train_df, sep='\t', names=['label', 'text'])
        wf = WordFilter()
        z_map = {}
        for z in zs:
            filter_words = wf.idf_filterset(original_df.text, type='automatic', z=z)
            save_filterset_tofile(filter_words, os.path.join(output_dir, 'z'+str(z)+'_filterset.txt'))
            z_map[z] = filter_words
        return z_map


def create_filterset(p2_raw_train_df, output_dir):
        original_df = pd.read_csv(p2_raw_train_df, sep='\t', names=['label', 'text'])
        wf = WordFilter()
        
        filter_words = wf.idf_filterset(original_df.text, type='manual', z=None)
        save_filterset_tofile(filter_words, os.path.join(output_dir, 'manual_filterset.txt'))
        return filter_words
