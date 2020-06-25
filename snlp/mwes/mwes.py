class MWE(object):
    """ Multiword Expressions (MWEs) class provides functionalities for an unsupervised extraction of MWEs from text.

    Features 
    TODO: 
    
    1. Functions extract MWEs from training data via PMI, NPMI, and SDMA from training corpus with application of POS tags keeping meaningful sequences (options: keep only NNNNs, JJNN, 
    or keeping all grammatically meaningful)
    2. Function to hyphenate the above MWEs in the corpus before creation of the embeddings so that MWEs get an embedding.
    3. If 2 is not applied, function to create summed embeddings for MWEs extracted in 1, and include them in fasttext/or other we model. 
    4. Function to find above MWEs (extracted in 1) in test data and hyphenate them. 
    """
    
    def __init__(self):
        pass
