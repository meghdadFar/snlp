import re

from nltk import word_tokenize
from tqdm import tqdm


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
