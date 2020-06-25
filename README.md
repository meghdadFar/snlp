# snlp

[![HitCount](http://hits.dwyl.com/meghdadFar/snlp.svg)](http://hits.dwyl.com/meghdadFar/snlp)

Statistical NLP (SNLP): A practical package with statisical natural language processing tools. SNLP is based on statistical and distributional attributes of natural language and hence most of the functionalities are unsupervised.

## Features
- Text cleaning 
- Text analysis
- Extraction of Fixed (Idiosyncratic) Expressions
- Identification of statistically redundant words for filtering. 

### Upcoming Features
- Anamoly Detection. 
- Identifying non-compositional compouds such as *red tape* and *brain drain* in the corpus .

## Usage

Install the package:

`pip3 install snlp`

See the description of different functionalities with worked examples below. 

### Text Cleaning

*snlp* implements an easy to use and powerful function for cleaning up the text (`clean_text`). 
Using, `clean_text`, you can choose what pattern to accept via `regex_pattern` argument, 
what pattern to drop via `drop` argument, and what pattern to replace via `replace` argument. You can also specify the maximum length of tokens. 
Let's use the IMDB Sentiment Dataset as an example (The easiest way to acquire the dataset is via [torchtext datasets](https://torchtext.readthedocs.io/en/latest/datasets.html#imdb)). 


```python
from snlp.preprocess import clean_text

imdb_train = pd.read_csv('imdb_train.tsv', sep='\t', names=['label', 'text'])

# Let's only keep alphanumeric tokens as well as important punctuation marks:
regex_pattern='^[a-zA-Z0-9!.,?\';:$/_-]+$'

# In this corpus, one can frequently see HTML tags such as `< br / >`. So let's drop them:
drop={'<br / >', '< br >'}

# By skimming throw the text one can frequently see many patterns such as !!! or ???. Let's replace them:
replace={'!!!':'!', '???':'?'}

# Finally, let's set the maximum length of a token to 15:
maxlen=15

imdb_train.text = imdb_train.text.apply(clean_text, args=('^[a-zA-Z0-9!.,?\';:$/_-]+$', 
                                                         {'<br / >', '< br >'}, 
                                                         {'!!!':'!', '???':'?'}, 
                                                         15,))
```

`clean_text` returns a tokenized text. 

### Text Analysis

*snlp* provides an easy to use function (`text_analysis.generate_report`) for analyzing text with an extensive analysis report. `text_analysis.generate_report` 
receives as input a dataframe that contains a text column, and an optional number of label columns. `text_analysis.generate_report` can generate plots for upto 4
numerical or categorical labels. See the example below for more details. 

```python
from snlp.text_analysis import generate_report

generate_report(df=imdb_train,
                out_dir='output_dir',
                text_col='text',
                label_cols=[('label', 'categorical')])

```

The above script creates an analysis report that includes distribution plots and word clouds for different POS tags, for text, and bar plots and histograms for labels. You can specify upto 
4 labels of type *categorical* or *numerical*. See the example below for including another label of *numerical* type. The report is automatically rendered in the browser via `plotly` default port assignment. But you also have the option of saving the report in an HTML format by setting the `save_report` argument to `True`. 

```python
import numpy as np
imdb_train['fake_label'] = np.random.randint(1, 500, imdb_train.shape[0])

generate_report(df=imdb_train,
                out_dir='output_dir',
                text_col='text',
                label_cols=[('label', 'categorical'), ('fake_label', 'numerical')])

```

### Extraction of Fixed (Idiosyncratic) Expressions

Identifying fixed expressions has application in a wide range of NLP taska ranging from sentiment analysis to topic models and keyphrase extraction. Fixed expressions are those multiword units whose components cannot be replaced with their near synonyms. E.g. *swimming pool* that cannot be replaced with *swim pool* or *swimmers pool*. 

You can use `snlp` to identify fixed noun-noun and adjective-nount expressions in your text leveraging statistical measures such as *PMI* and *NPMI*. To do so, first import required libraries: 

```python
from snlp.mwes import get_counts, get_ams

imdb_train = imdb_train.text.apply( lambda x : x.lower())
```

Run `get_counts` to extract compounds and their corresponding frequencies and then run `get_ams` to calculate their corresponding *PMI* and rank them based on their *PMI* value:

```python
get_counts(imdb_train, text_column='text', output_dir='tmp/')
get_ams(path_to_counts='tmp/')
```

The ranked compounds can be found in `output_dir`. 


