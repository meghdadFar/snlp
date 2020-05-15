# snlp

[![HitCount](http://hits.dwyl.com/meghdadFar/snlp.svg)](http://hits.dwyl.com/meghdadFar/snlp)

Statistical NLP (SNLP): A practical package with statisical natural language processing tools. SNLP is based on statistical and distributional attributes of natural language and hence most of the functionalities are unsupervised.

## Features
- Identifying Multiword Expressions (Collocations) in the corpus. Used for terminology and keyphrase extraction. Can lead to improvement in text classification. 
- Identifying statistically redundant words for filtering. Usually leads to an improvement in document classification. 

### Upcoming Features
- Anamoly Detection. 
- Identifying non-compositional compouds: Can be used for tasks such as profanity/hate-speech detection, and linguistic analysis of a corpus.

## Usage

First install the package:

`pip3 install snlp`

Below you can find worked explames for various usages of the package. 

### Extract fixed (idiosyncratic) Noun-Noun and Adj-Noun Compounds

Identifying fixed expressions has application in a wide range of NLP taska ranging from sentiment analysis to topic models and keyphrase extraction. Fixed expressions are those multiword units whose components cannot be replaced with their near synonyms. E.g. *swimming pool* that cannot be replaced with *swim pool* or *swimmers pool*. 

You can use `snlp` to identify fixed noun-noun and adjective-nount expressions in your text leveraging statistical measures such as *PMI* and *NPMI*. To do so, first import required libraries: 

```
import snlp
import pandas as pd

from snlp.mwes.am import get_counts, get_ams
from nltk.tokenize import word_tokenize
from essential_generators import DocumentGenerator
```
Install `essential_generators` using pip in order to generate random text. In the following example, you also need nltk for some preprocessing. `nltk` is already installed as a dependency of `snlp`. So you don't need to install it again. 

Then create a dataframe and populate it with random text:

```
mydf = pd.DataFrame(columns=['text', 'topic'])
gen = DocumentGenerator()
mydf.text = [gen.sentence() for i in range(10)]
mydf.topic = [gen.word() for i in range(10)]
```

Do some preprocessing:

mydf.text = mydf.text.apply(word_tokenize).apply(lambda x : ' '.join(x)).apply(lambda x : x.lower())
mydf.topic = mydf.topic.apply( lambda x : x.lower())
```

Run `get_counts` to extract required compounds, their corresponding frequencies and then `get_ams` for the calculation of *PMI* and ranking compounds based on their *PMI* value:

```
get_counts(mydf, text_column='text', output_dir='tmp/')
get_ams(path_to_counts='tmp/')
```

The results can be found in `output_dir` (which is the same as `path_to_counts`). 


