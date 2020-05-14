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
        """Plots word scores/frequency needed for different types of analysis.
    
        """
        vectorizer = TfidfVectorizer(min_df=1)
        X = vectorizer.fit_transform(documents)

    def idf_filterset(self, documents, type='automatic', z=None):
        """Creates a filter set.

        Parameters
        ----------
        documents : iterable
            An iterable which yields either str, unicode or file objects.
        type : string
            Type of filterset creation: automatic or manual.

        Returns
        -------
        filter_set : 
            This is filterset
        """
        vectorizer = TfidfVectorizer(min_df=1)
        X = vectorizer.fit_transform(documents)
        
        if type == 'automatic':
            if z:
                filterset = self._create_filter_zscore(vectorizer, zscore_threshold=z)
            else:
                filterset = self._create_filter_zscore(vectorizer)
        elif type == 'manual':
            filterset = self._create_filter(vectorizer, lower_idf=1.5, upper_idf=11)

        return filterset
        
    def _create_filter(self, vectorizer, lower_idf, upper_idf):
        """

        Parameters
        ----------
        lower_idf (inclusive)
        upper_idf (inclusive)

        """
        idf = vectorizer.idf_
        tfidf = dict(zip(vectorizer.get_feature_names(), idf))
        sl = sorted(tfidf.items(), key=lambda kv: kv[1])

        smallest_idf = sl[0][1]
        largest_idf = sl[len(sl)-1][1]

        # if lower_idf < smallest_idf:
        #     logger.error("Idf values are between [%.2f, %.2f]. You have set lower_idf to %.2f. Continuing with automatic filter creation." 
        #                 %(smallest_idf, largest_idf, lower_idf))
        #     filterset = self._create_filter_zscore(vectorizer)
        #     return filterset
            
        # if upper_idf > largest_idf:
        #     logger.error("Idf values are between [%.2f, %.2f]. You have set upper_idf to %.2f. Continuing with automatic filter creation." 
        #                 %(smallest_idf, largest_idf, upper_idf))
        #     filterset = self._create_filter_zscore(vectorizer)
        #     return filterset

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
        idf_df = pd.DataFrame(token_idp_dict.items(), columns=['token', 'idf'])

        # Gaussianize idf
        g = gaussianize.Gaussianize(strategy='brute')
        g.fit(idf_df['idf'])
        idf_guassian = g.transform(idf_df['idf'])
        idf_df['idf_gaussianized'] = idf_guassian

        # Calculate z score
        z = zscore(idf_df['idf_gaussianized'])
        idf_df['zscore'] = z

        filter_set = set()
        for i in range(len(idf_df)):
            if idf_df.iloc[i]['zscore'] <= -zscore_threshold or idf_df.iloc[i]['zscore'] >= zscore_threshold:
                filter_set.add(idf_df.iloc[i]['token'])
        return filter_set


    def _create_subplots(self, scores, titles, path_to_plot):
        """ Creates subplots corresponding to the histogram of scores. 

        Parameters
        ----------
        scores list of list 

        """

        cmap = plt.cm.get_cmap('hsv', len(scores))
        fig, ax = plt.subplots(1, len(scores))
        for i in range(len(scores)):
            ax[i].set_title(titles[i])
            ax[i].hist(scores[i], alpha=0.7, color=cmap(i))

        fig.savefig(path_to_plot)