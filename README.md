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
Let's use [Stanford's IMDB Sentiment Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) as an example. A sample of this data can be found in `resources/data/imdb_train_sample.tsv`.


```python
from snlp.preprocessing import clean_text

imdb_train = pd.read_csv('resources/data/imdb_train_sample.tsv', sep='\t', names=['label', 'text'])

# Let's only keep alphanumeric tokens as well as important punctuation marks:
regex_pattern='^[a-zA-Z0-9!.,?\';:$/_-]+$'

# In this corpus, one can frequently see HTML tags such as `< br / >`. So let's drop them:
drop={'< br / >'}

# By skimming throw the text one can frequently see many patterns such as !!! or ???. Let's replace them:
replace={'!!!':'!', '\?\?\?':'?'}

# Finally, let's set the maximum length of a token to 15:
maxlen=15

imdb_train.text = imdb_train.text.apply(clean_text, args=(regex_pattern, drop, replace, maxlen,))
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
import random

imdb_train['numerical_label'] = np.random.randint(1, 500, imdb_train.shape[0])
imdb_train['new_label'] = random.choices(['a', 'b', 'c', 'd'], [0.2, 0.5, 0.8, 0.9], k=imdb_train.shape[0])

generate_report(df=imdb_train,
                out_dir='output_dir',
                text_col='text',
                label_cols=[('label', 'categorical'), ('new_label', 'categorical'), ('numerical_label', 'numerical')])

```

The above yields a report in HTML, with interactive `plotly` plots as can be seen in example screenshots below. 

![annotation1](https://github.com/meghdadFar/snlp/blob/master/resources/images/annotation1.png)
![text](https://github.com/meghdadFar/snlp/blob/master/resources/images/text.png)
![toolbar](https://github.com/meghdadFar/snlp/blob/master/resources/images/toolbar.png)
![zoom](https://github.com/meghdadFar/snlp/blob/master/resources/images/zoom.png)
![labels](https://github.com/meghdadFar/snlp/blob/master/resources/images/labels.png)
![wc](https://github.com/meghdadFar/snlp/blob/master/resources/images/wc.png)


### Extraction of Fixed (Idiosyncratic) Expressions

Identifying fixed expressions has application in a wide range of NLP taska ranging from sentiment analysis to topic models and keyphrase extraction. Fixed expressions are those multiword units whose components cannot be replaced with their near synonyms. E.g. *swimming pool* that cannot be replaced with *swim pool* or *swimmers pool*. 

You can use `snlp` to identify fixed noun-noun and adjective-nount expressions in your text leveraging statistical measures such as *PMI* and *NPMI*. To do so, first import required libraries: 
Run `get_counts` to extract compounds and their corresponding frequencies and then run `get_ams` to calculate their corresponding *PMI* and rank them based on their *PMI* value:

```python
from snlp.mwes import get_counts, get_ams

get_counts(imdb_train, text_column='text', output_dir='tmp/')
get_ams(path_to_counts='tmp/')
```

Running the above yields two sets of ranked noun-noun and adjective-noun expressions that can be found in `output_dir` under `nn_pmi.json` and `jn_pmi.json`. Some examples from the top of ranked fixed expressions can be seen below:

```
nn_pmi.json:
[[busby berkeley", [11.2, 0.94]], ["burgess meredith", [11.13, 0.92]], ["jet li", [10.0, 0.89]], 
["clint eastwood", [10.27, 0.89]], ["claire danes", [10.06, 0.89]], ["bruno mattei", [10.92, 0.89]], 
["kung fu", [9.55, 0.88]], ["monty python", [10.69, 0.86]], ["denise richards", [10.63, 0.86]], ["victor mclaglen", [10.33, 0.85]], 
["blade runner", [10.48, 0.85]], ["guinea pig", [10.52, 0.85]]

jn_pmi.json:
[["spinal tap", [11.35, 0.96]], ["uwe boll", [11.0, 0.96]], ["martial arts", [9.52, 0.95]],
["farrah fawcett", [11.2, 0.93]], ["eli wallach", [11.14, 0.93]], ["akshay kumar", [10.7, 0.9]], 
["timothy dalton", [10.04, 0.89]], ["rita hayworth", [10.85, 0.89]], ["citizen kane", [10.11, 0.88]], ["facial expressions", [10.06, 0.88]], 
["united states", [9.72, 0.88]], ["global warming", [10.44, 0.87]]
```

