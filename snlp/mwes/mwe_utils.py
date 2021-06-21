import json
import collections
import pandas
import re
import tqdm


def replace_compunds(path_to_compounds, df, text_column, am_threshold=0.5, only_compounds=False, lower_case=False):
    """Hyphenates the compounds in the corpus so that they are treated as a single token by downstream applications.

    Args:
        
        path_to_nns (string): Path to the ranked list of compounds
        df (pd.FataFrame): DataFrame comprising training data with a text column
        text_column (string): Text (content) column of df
        am_threshold (float): AM threshold above which the compounds are selected for replacement
        only_compounds (bool): Whether or not keep only compounds and drop the rest of the text

    Returns:
        df (pandas.FataFrame)

    """

    json_file = open(path_to_compounds)
    json_str = json_file.read()

    pmi_ord_dict = json.loads(json_str, object_pairs_hook=collections.OrderedDict)
    print('Num of all compounds: %d' %len(pmi_ord_dict))
    good_nns = set()
    
    # Current format of pmi_ord_dict: [['victor jones', [14.72, 0.96]], ...]
    # pmi_ord_dict[i][1][1]: npmi  | pmi_ord_dict[i][1][0]: pmi   |  pmi_ord_dict[i][0]: compound
    i = 0
    while  float(pmi_ord_dict[i][1][1]) > am_threshold:
        good_nns.add(pmi_ord_dict[i][0])
        i += 1
    
    print('Number of compounds to be replaced in corpus: %d' %len(good_nns))

    new_text = []
    for sent in tqdm.tqdm(df[text_column]):
        # Extract bigrams from text
        sent = (sent.lower() if lower_case else sent)
        bigrams = get_ngrams(sent, 2)
        
        if only_compounds:
            tmp = ''
            for bg in bigrams:
                if bg in good_nns:
                    tmp += (bg.split(' ')[0]+'-'+bg.split(' ')[1]+' ')
            if tmp != '':
                sent = tmp.strip()
        else:
            for bg in bigrams:
                if bg in good_nns:
                    sent = re.sub(bg, bg.split(' ')[0]+'-'+bg.split(' ')[1], sent)
        new_text.append(sent)
    df[text_column] = new_text
    return df


def get_ngrams(sentence, n):
    """Extracts n-grams from sentence.
    
    Args:
        sentence (string)
        n (int)
        lower(bool)

    Returns:
        ngrams(list)
    """
    #TODO handle excecases when sent is nan, here and in the calling function
    tokens = sentence.split(" ")
    ngrams = []
    for i in range(len(tokens)-n+1):
        ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams