import nltk
import tqdm
import json
import math
import re
import os

import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.tokenize import ToktokTokenizer
from collections import Counter
from snlp import logger 


"""
TODO: 
    
    1. Include more AMs
    2. Include SDMA
    3. Function to create embedding for MWEs by combining the embeding of their components. 
"""


def extract_ncs_from_sent(sentence):
    """Extract two-word noun compounds from tokenized input. 
    
    Args:
        sentence (string): A tokenized sentence. 

    Returns:
        NNS_counts (dict): Dictionary of noun-noun compounds to their count in sentence. 
        JNs_counts (dict): Dictionary of adjective-noun compounds to their count in sentence. 
        word_counts (dict) Dictionary of words to their count in sentence. 
    """
    if not isinstance(sentence, str):
        raise TypeError('Input must be a string.')
    
    if len(sentence) == 0:
        return
        
    tokens = sentence.split(" ")
    postag_tokens = nltk.pos_tag(tokens)
    JNs = []
    NNs = []
    for i in range(len(postag_tokens)-1):
        w1 = postag_tokens[i]
        if w1[1] not in ['NN', 'NNS', 'JJ']:
            continue
        else:
            w2 = postag_tokens[i+1]
            if not re.match("[a-zA-Z0-9]{2,}", w1[0]) or not re.match("[a-zA-Z0-9]{2,}", w2[0]):
                continue

            if w2[1] in ['NN', 'NNS']:
                if i+2 < len(postag_tokens):
                    w3 = postag_tokens[i+2]
                    if w3 not in ['NN', 'NNS']:
                        if w1[1] == 'NN':
                            NNs.append(w1[0]+' '+w2[0])
                        if w1[1] == 'JJ':
                            JNs.append(w1[0]+' '+w2[0])
                else:
                    if w1[1] == 'NN':
                        NNs.append(w1[0]+' '+w2[0])
                    if w1[1] == 'JJ':
                        JNs.append(w1[0]+' '+w2[0])
    
    NNS_counts = Counter(NNs) 
    JNs_counts = Counter(JNs) 
    word_counts = Counter(tokens)
    return NNS_counts, JNs_counts, word_counts


def calculate_pmi(nn_dict, word_dic, N_compounds, N_words):
    """Calculates Pointwise Mutual Information between the two words of every word pair in nn_dict. 

    Args:
        nn_dict (dict): Dictionary of compounds and their count. 
        word_dic (dict): Dictionary of words and their count. 
        N_compounds (int): Number of compounds.
        N_words (int): Number of words. 
    
    Returns:
        nn_dict (dict): Dictionary of compounds and their pmi and npmi values, sorted based on their npmi. 

    """
    pmi = 0.0
    npmi = 0.0
    for compound, count in nn_dict.items():
        w1w2 = compound.split(" ")
        # To filter out compounds that are rare/unique because of strange/misspelled component words. 
        if float(word_dic[w1w2[0]]) > 10 and float(word_dic[w1w2[1]]) > 10:
            p_of_c = float(count) / float(N_words)
            p_of_h = float(word_dic[w1w2[0]]) / float(N_words)
            p_of_m = float(word_dic[w1w2[1]]) / float(N_words)

            pmi = math.log(p_of_c/(p_of_h * p_of_m))
            npmi = pmi/float(-math.log(p_of_c))
            nn_dict[compound] = [round(pmi, 2), round(npmi, 2)]
        else:
            nn_dict[compound] = [0.0, 0.0]

    nn_dict = sorted(nn_dict.items(), key=lambda e: e[1][1], reverse=True)
    return nn_dict


def get_counts(df, text_column, output_dir):
    """Reads a corpus in df (pd.DataFrame) and generates all counts necessary for calculating AMs.

    Args:
        output_dir (string): Path to output dir where calculated counts will be saved. 

    Returns:
        0 (int): If count files are successfully stored in the specified path. 
    """
    NNs_total = {}
    JNs_total = {}
    W_total = {}

    NN_count = 0
    JN_count = 0
    W_count = 0

    for sent in tqdm.tqdm(df[text_column]):
        NNs, JNs, Ws = extract_ncs_from_sent(sent)
        
        for k,v in NNs.items():
            NN_count += v
            if k in NNs_total:
                NNs_total[k] += v
            else:
                NNs_total[k] = v
        
        for k, v in JNs.items():
            JN_count += v
            if k in JNs_total:
                JNs_total[k] += v
            else:
                JNs_total[k] = v
        
        for k, v in Ws.items():
            W_count += v
            if k in W_total:
                W_total[k] += v
            else:
                W_total[k] = v

    try:
        with open(os.path.join(output_dir, 'nns.json'), 'w') as file:
            file.write(json.dumps(NNs_total))

        with open(os.path.join(output_dir, 'jns.json'), 'w') as file:
            file.write(json.dumps(JNs_total))

        with open(os.path.join(output_dir, 'words.json'), 'w') as file:
            file.write(json.dumps(W_total))

        with open(os.path.join(output_dir, 'jn_count.txt') , 'w') as file:
            file.write(str(JN_count))
        
        with open(os.path.join(output_dir, 'nn_count.txt'), 'w') as file:
            file.write(str(NN_count))

        with open(os.path.join(output_dir, 'w_count.txt'), 'w') as file:
            file.write(str(W_count))
    except:
        raise
    else:
        return 0



def get_ams(path_to_counts, ams=['pmi']):
        """Reads the counts from path_to_counts and accordingly, calculates measures specified in ams.
        
        Args:
            path_to_counts (string): Path to counts dir, generated by get_counts. 
            ams (list): List of AMs to be calcuated. 
        
        Returns:
        
        """
        f = open(os.path.join(path_to_counts, 'w_count.txt'))
        v = f.read()
        w_count = int(v)

        f = open(os.path.join(path_to_counts, 'nn_count.txt'))
        v = f.read()
        nn_count = int(v)

        f = open(os.path.join(path_to_counts, 'jn_count.txt'))
        v = f.read()
        jn_count = int(v)

        json_file = open(os.path.join(path_to_counts, 'jns.json'))
        json_str = json_file.read()
        jns = json.loads(json_str)

        json_file = open(os.path.join(path_to_counts, 'nns.json'))
        json_str = json_file.read()
        nns = json.loads(json_str)

        json_file = open(os.path.join(path_to_counts, 'words.json'))
        json_str = json_file.read()
        words = json.loads(json_str)

        if 'pmi' in ams:
            res = calculate_pmi(nn_dict=nns, word_dic=words, N_compounds=nn_count, N_words=w_count)
            with open(os.path.join(path_to_counts, 'nn_pmi.json'), 'w') as file:
                file.write(json.dumps(res))

            res = calculate_pmi(nn_dict=jns, word_dic=words, N_compounds=jn_count, N_words=w_count)
            with open(os.path.join(path_to_counts, 'jn_pmi.json'), 'w') as file:
                file.write(json.dumps(res))
        
        if 'chis' in ams:
            print('Chis is not implemented.')
        
        if 't' in ams:
            print('t is not implemented.')


    