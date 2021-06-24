# snlp

[![PyPI version](https://badge.fury.io/py/snlp.svg?&kill_cache=1)](https://badge.fury.io/py/snlp)

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

[![HitCount](http://hits.dwyl.com/meghdadFar/snlp.svg)](http://hits.dwyl.com/meghdadFar/snlp)


Statistical NLP (SNLP) is a practical package with statisical tools for natural language processing. SNLP is based on statistical and distributional attributes of natural language and hence most of its functionalities are unsupervised. 

## Features
- Text cleaning 
- Text analysis
- Extraction of Fixed (Idiosyncratic) Expressions
- Identification of statistically redundant words for filtering

### Upcoming Features
- Anamoly Detection
- Identifying non-compositional compouds such as *red tape* and *brain drain* in the corpus

## Usage

Install the package:

`pip3 install snlp`

See the description of different functionalities with worked examples below. 

### **Text Cleaning**

*snlp* implements an easy to use and powerful function for cleaning up the text (`clean_text`). 
Using, `clean_text`, you can choose what pattern to accept via `keep_pattern` argument, 
what pattern to drop via `drop_patterns` argument, and what pattern to replace via `replace` argument. You can also specify the maximum length of tokens. 
Let's use [Stanford's IMDB Sentiment Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) as an example. A sample of this data can be found in `resources/data/imdb_train_sample.tsv`.


```python
from snlp.preprocessing import clean_text

imdb_train = pd.read_csv('resources/data/imdb_train_sample.tsv', sep='\t', names=['label', 'text'])

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

`clean_text` returns a tokenized text. 

### **Text Analysis**

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

![annotation1](https://github.com/meghdadFar/snlp/blob/master/resources/images/annotation1.png)
<!-- ![text](https://github.com/meghdadFar/snlp/blob/master/resources/images/text.png) -->
![toolbar](https://github.com/meghdadFar/snlp/blob/master/resources/images/toolbar.png)
![zoom](https://github.com/meghdadFar/snlp/blob/master/resources/images/zoom.png)
<!-- ![labels](https://github.com/meghdadFar/snlp/blob/master/resources/images/labels.png) -->
![wc](https://github.com/meghdadFar/snlp/blob/master/resources/images/wc.png)


### **Extraction of Fixed (Idiosyncratic) Expressions**

Identifying fixed expressions has application in a wide range of NLP taska ranging from sentiment analysis to topic models and keyphrase extraction. Fixed expressions are those multiword units whose components cannot be replaced with their near synonyms. E.g. *swimming pool* that cannot be replaced with *swim pool* or *swimmers pool*. 

You can use `snlp` to identify fixed noun-noun and adjective-nount expressions in your text leveraging statistical measures such as *PMI* and *NPMI*. To do so, first import `get_counts` and `get_ams` functions: 
Run `get_counts` to extract compounds and their corresponding frequencies and then run `get_ams` to calculate their corresponding *PMI* and rank them based on their *PMI* value:

```python
from snlp.mwes import get_counts, get_ams

get_counts(imdb_train, text_column='text', output_dir='tmp/')
get_ams(path_to_counts='tmp/')
```

Running the above yields two ranked sets of  *noun-noun* and *adjective-noun* compounds that can be found in `output_dir` respectively under `nn_pmi.json` and `jn_pmi.json`. Some examples from the top of ranked compounds can be seen below:

```
nn_pmi.json
-----------
jet li
clint eastwood
monty python
kung fu
blade runner


jn_pmi.json
-----------
spinal tap
martial arts
citizen kane
facial expressions
global warming
```

The main idea behind the extraction of fixed Expressions is to treat them as a single token. Research shows that when fixed expressions are treated as a single token rather than the sum of their components, they can improve the performance of downstream applications such as classification and NER. Using `snlp.mwe.replace_compunds` function, you can replace the extracted expressions in the corpus with their hyphenated version (global warming --> global-warming) so that they are considered a single token by downstream appilcations. 

### **Identification of Statistically Redundant Words**

Redundant words carry little value and can exacerbate the results of many NLP tasks. To solve this issue, traditionally, a pre-defined list of words, called stop words was defined and removed from the data. However, creating such a list is not optimal because in addition to being a rule-based and manual approach which does not generalize well, one has to assume that there is a uneversal list of stop words that represents highly low entropy words for all corpora, which is a very strong assumption and not necessarily a true assumption in many cases.

To solve this issue, one can use a purely sttistical solution which is completely automatic and does not make any universal assumption. It only focuses on the corpus at hand. Words can be represented with various statistics. For instance, they can be represented by term frequency (tf) or inverse document ferquency (idf). Terms with anomalous (very high or very low) statistics usually carry little value.
SNLP enables you to identify such terms in a completely automatic fashion. While the solution is complex behind the scene, as it firsts needs to gaussanizing the distribution of specified statistics (i.e. tf or ifd), and then identifying words with anomalous values on the gaussanized distribution by looking at their z-score, the API is very easy and convinient to use. The example below shows how you can use this API:




