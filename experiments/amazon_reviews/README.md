# Experiments on Amazon Product Review Dataset

## Usage

1. Create update a `amazon_review.json config file in snlp/config
2. `python experiments/amazon_reviews/mwe_amazon_prep_data.py`
3. `python experiments/amazon_reviews/mwe_amazon_getcounts.py`
4. `python experiments/amazon_reviews/mwe_amazon_getpmi.py`
5. `python experiments/amazon_reviews/mwe_amazon_repl.py`
6. `python experiments/amazon_reviews/mwe_amazon_train_test_split.py`
7. `python experiments/amazon_reviews/mwe_amazon_train.py`
8. `python experiments/amazon_reviews/mwe_amazon_eval.py`

Running the above script leads to logging the evlauation results as well as corresponding confusion matrices. 