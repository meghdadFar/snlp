# snlp

[![PyPI version](https://badge.fury.io/py/snlp.svg?&kill_cache=1)](https://badge.fury.io/py/snlp)

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)


Statistical NLP (SNLP) is a practical package with statisical tools for natural language processing. SNLP is based on statistical and distributional attributes of natural language and hence most of its functionalities are unsupervised. 

# Features
- [Text Cleaning](#text-cleaning)
- [Text Analysis](#text-analysis)
- [Extraction of Multiword Expressions](#extraction-of-multiword-expressions)
- [Identification of Statistically Redundant Words](#identification-of-statistically-redundant-words)

## Upcoming Features
- Identification of non-compositional expressions such as *red tape* and *brain drain*

# Usage

Install the package:

`pip install snlp`

See the description of different functionalities with worked examples below. 

## **Text Cleaning**

*snlp* implements an easy to use and powerful function for cleaning up the text (`clean_text`). 
Using, `clean_text`, you can choose what pattern to accept via `keep_pattern` argument, 
what pattern to drop via `drop_patterns` argument, and what pattern to replace via `replace` argument. You can also specify the maximum length of tokens. 
Let's use [Stanford's IMDB Sentiment Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) as an example. A sample of this data can be found in `resources/data/imdb_train_sample.tsv`.


```python
from snlp.preprocessing import clean_text

imdb_train = pd.read_csv('data/imdb_train_sample.tsv', sep='\t', names=['label', 'text'])

# Let's only keep alphanumeric tokens as well as important punctuation marks:
keep_pattern='^[a-zA-Z0-9!.,?\';:$/_-]+$'

# In this corpus, one can frequently see HTML tags such as `< br / >`. So let's drop them:
drop_patterns={'< br / >'}

# By skimming throw the text one can frequently see many patterns such as !!! or ???. Let's replace them:
replace={'!!!':'!', '\?\?\?':'?'}

# Finally, let's set the maximum length of a token to 15:
maxlen=15

imdb_train.text = imdb_train.text.apply(clean_text, args=(keep_pattern, drop_patterns, replace, maxlen,))
```

Note that `clean_text` returns tokenized text. 

## **Text Analysis**

*snlp* provides an easy to use function (`text_analysis.generate_report`) for analyzing text with an extensive analysis report. `text_analysis.generate_report` 
receives as input a dataframe that contains a text column, and an optional number of label columns. Currently, `text_analysis.generate_report` can generate plots for upto 4
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
import random

# In addition to the original label, for illustration purpose, let's create two random labels:
imdb_train['numerical_label'] = np.random.randint(1, 500, imdb_train.shape[0])
imdb_train['new_label'] = random.choices(['a', 'b', 'c', 'd'], [0.2, 0.5, 0.8, 0.9], k=imdb_train.shape[0])

generate_report(df=imdb_train,
                out_dir='output_dir',
                text_col='text',
                label_cols=[('label', 'categorical'), ('new_label', 'categorical'), ('numerical_label', 'numerical')])

```

The above yields a report in HTML, with interactive `plotly` plots as can be seen in example screenshots below. 

![annotation1](/assets/annotation1.png)

 You can easily zoom in any part of the plot to a have a closer look:

![zoom](/assets/zoom.png)

You can get word clouds for different part of speech tags, as can be seen in the below example where word clouds for nouns, adjectives and verbs are rendered:

![wc](/assets/wc.png)

## **Extraction of Multiword Expressions**

Identifying fixed expressions has application in a wide range of NLP taska ranging from sentiment analysis to topic models and keyphrase extraction. Fixed expressions are those multiword units whose components cannot be replaced with their near synonyms. E.g. *swimming pool* that cannot be replaced with *swim pool* or *swimmers pool*. 

You can use `snlp` to identify different types of MWEs in your text leveraging statistical measures such as *PMI* and *NPMI*. To do so, first create an instance of `MWE` class:


```python
from snlp.mwes import MWE
my_mwe_types = ["NC", "JNC"]
mwe = MWE(df=imdb_train, mwe_types=my_mwe_types, text_column='text')
```

If the text in `text_column` is untokenized or poorly tokenized, `MWE` recognizes this issue at instantiation time and shows you a warning. If you already know that your text is not tokenized, you can run the same instantiation with flag `tokenize=True`. Next you need to run the method `build_count()`. Since creating counts is a time consuming procesure, it was implemented independently from `extract_mwes()` method that works on top of the output of `build_count()`. This way, you can get the counts which is a time consuming process once, and then run `extract_mwes()` several times with different parameters.

```python
mwe.build_counts()
mwe.extract_mwes()
```

Running the above results in a json file, containing dictionary of mwe types defined in the `mwe_types` argument of `MWE`, to their association score (specified by `am` argument of `extract_mwes()`). Note that the MWEs in this json file are sorted with respect to their `am` score. All MWEs and their counts are stored in respective directories inside the `output_dir` argument of `MWE`. The default value is `tmp`. 

```
NOUN-NOUN COMPOUNDS
-------------------
jet li
clint eastwood
monty python
kung fu
blade runner


ADJECTIVE-NOUN COMPOUNDS
------------------------
spinal tap
martial arts
citizen kane
facial expressions
global warming
```

An important use of extracting MWEs is to treat them as a single token. Research shows that when fixed expressions are treated as a single token rather than the sum of their components, they can improve the performance of downstream applications such as classification and NER. Using the `replace_mwes` function, you can replace the extracted expressions in the corpus with their hyphenated version (global warming --> global-warming) so that they are considered a single token by downstream appilcations. A worked example can be seen below:

```python
from snlp.mwes import replace_mwes
new_df = replace_mwes(path_to_mwes='tmp/mwes/mwe_data.json', mwe_types=['NC', 'JNC'], df=imdb_train, text_column='text')
new_df.to_csv('tmp/new_df.csv', sep='\t')
```


## **Identification of Statistically Redundant Words**

Redundant words carry little value and can exacerbate the results of many NLP tasks. To solve this issue, traditionally, a pre-defined list of words, called stop words was defined and removed from the data. However, creating such a list is not optimal because in addition to being a rule-based and manual approach which does not generalize well, one has to assume that there is a uneversal list of stop words that represents highly low entropy words for all corpora, which is a very strong assumption and not necessarily a true assumption in many cases.

To solve this issue, one can use a purely sttistical solution which is completely automatic and does not make any universal assumption. It focuses only on the corpus at hand. Words can be represented with various statistics. For instance, they can be represented by their term frequency (tf) or inverse document ferquency (idf). It can be then interpreted that terms with anomalous (very high or very low) statistics carry little value and can be discardrd.
SNLP enables you to identify such terms in an automatic fashion. The solution might seem complex behind the scene, as it firsts needs calculate certain statistics, gaussanize the distribution of the specified statistics (i.e. tf or ifd), and then identify the terms with anomalous values on the gaussanized distribution by looking at their z-score. However, the API is easy and convinient to use. The example below shows how you can use this API:

```python
from snlp.preprocessing import RedunWords

imdb_train = pd.read_csv('resources/data/imdb_train_sample.tsv', sep='\t', names=['label', 'text'])
rw = RedunWords(imdb_train["text"], method='idf')
```

Let the program automatically identify a set of redundant words:

```python
red_words = rw.get_redundant_terms()
```


Alternatively, you can manually set cut-off threshold for the specified score, by setting the manual Flag to True and specifying lower and upper cut-off thresholds. 
```python
red_words = rw.get_redundant_terms(manual=True, manual_thresholds: dict={'lower_threshold':1, 'upper_threshold': 8})
```

In order to get a better undertanding of the distribution of the scores before setting the thresholds, you can run `show_plot()` method from `RedunWords` class to see this distribution:

```python
rw.show_plot()
```

When red_words is ready, you can filter the corpus:

```python
# text must be a list of words
res = " ".join([t for t in text if t not in redundant_terms])
```


