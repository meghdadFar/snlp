import os

import pandas as pd

from matplotlib import pyplot as plt
from scipy.stats import zscore
from snlp import logger
from snlp import gaussianize
from sklearn.feature_extraction.text import TfidfVectorizer


class WordFilter(object):
    def __init__(self):
        pass

    def analysis_report(self, documents):
        """Plot word frequency needed for different types of analysis.

        """
        vectorizer = TfidfVectorizer(min_df=1)
        X = vectorizer.fit_transform(documents)

    def idf_filterset(self, documents, method="automatic", z=None, l_idf=1, u_idf=8):
        """Creates a filter set by identifying words with anomalous IDF value.

        Args:
            documents (iterable): An iterable which yields either str, unicode or file objects.
            method (string): Method of creating filterset: automatic or manual.
            z (float): Z-score of the IDF distribution above which words are considered anomalous.
            l_idf  (float): Lower cut-off threshold for IDF (inclusive). Used only when 
            u_idf (float): Upper cut-off threshold for IDF (inclusive).

        Returns:
            filter_set (set): Set of filter words
        """

        vectorizer = TfidfVectorizer(min_df=1)
        X = vectorizer.fit_transform(documents)

        if method == "automatic":
            if z:
                filterset = self._create_filter_zscore(vectorizer, zscore_threshold=z)
            else:
                filterset = self._create_filter_zscore(vectorizer)
        elif method == "manual":
            filterset = self._create_filter(vectorizer, lower_idf=l_idf, upper_idf=u_idf)

        return filterset

    def _create_filter(self, vectorizer, lower_idf, upper_idf):
        """

        Args:
            lower_idf  (float): Lower cut-off threshold for IDF (inclusive).
            upper_idf (float): Upper cut-off threshold for IDF (inclusive).

        """

        idf = vectorizer.idf_
        tfidf = dict(zip(vectorizer.get_feature_names(), idf))
        sl = sorted(tfidf.items(), key=lambda kv: kv[1])

        smallest_idf = sl[0][1]
        largest_idf = sl[len(sl) - 1][1]

        if lower_idf < smallest_idf:
            raise ValueError(
                "Idf values are between [%.2f, %.2f]. You have set lower_idf to %.2f. Update the values accordingly, \
                or consider continuing with automatic filter creation."
                % (smallest_idf, largest_idf, lower_idf)
            )

        if upper_idf > largest_idf:
            raise ValueError(
                "Idf values are between [%.2f, %.2f]. You have set upper_idf to %.2f. Update the values accordingly, \
                or consider using automatic filter creation."
                % (smallest_idf, largest_idf, upper_idf)
            )

        filter_words = set()
        for i in range(len(sl)):
            if sl[i][1] < lower_idf:
                filter_words.add(sl[i][0])
            if sl[i][1] >= upper_idf:
                filter_words.add(sl[i][0])

        return filter_words

    def _create_filter_zscore(self, vectorizer, zscore_threshold=3):
        idf = vectorizer.idf_
        token_idp_dict = dict(zip(vectorizer.get_feature_names(), idf))
        idf_df = pd.DataFrame(token_idp_dict.items(), columns=["token", "idf"])

        # Gaussianize idf
        g = gaussianize.Gaussianize(strategy="brute")
        g.fit(idf_df["idf"])
        idf_guassian = g.transform(idf_df["idf"])
        idf_df["idf_gaussianized"] = idf_guassian

        # Calculate z score
        z = zscore(idf_df["idf_gaussianized"])
        idf_df["zscore"] = z

        filter_set = set()
        for i in range(len(idf_df)):
            if idf_df.iloc[i]["zscore"] <= -zscore_threshold or idf_df.iloc[i]["zscore"] >= zscore_threshold:
                filter_set.add(idf_df.iloc[i]["token"])
        return filter_set

    def _create_subplots(self, scores):
        """ Creates subplots corresponding to the histogram of scores. 
        """
        raise NotImplementedError


def filter_text(text, filter_set):
    """
    Args:
        text (list): tokenized text
        filterset (set): Set of filter words
    """

    if not isinstance(text, list):
        raise TypeError("Input must be a list.")
    if len(text) == 0:
        raise ValueError("Input must be a non empty list.")

    res = " ".join([t for t in text if t not in filter_set])
    return res


def save_filterset_tofile(filter_set, path):
    with open(path, "w") as f:
        for word in filter_set:
            f.write(word + "\n")
    f.close()


def create_filterset_map(p2_raw_train_df, output_dir, zs=[3]):
    original_df = pd.read_csv(p2_raw_train_df, sep="\t", names=["label", "text"])
    wf = WordFilter()
    z_map = {}
    for z in zs:
        filter_words = wf.idf_filterset(original_df.text, method="automatic", z=z)
        save_filterset_tofile(filter_words, os.path.join(output_dir, "z" + str(z) + "_filterset.txt"))
        z_map[z] = filter_words
    return z_map


def create_filterset(p2_raw_train_df, output_dir):
    original_df = pd.read_csv(p2_raw_train_df, sep="\t", names=["label", "text"])
    wf = WordFilter()

    filter_words = wf.idf_filterset(original_df.text, method="manual", z=None)
    save_filterset_tofile(filter_words, os.path.join(output_dir, "manual_filterset.txt"))
    return filter_words
