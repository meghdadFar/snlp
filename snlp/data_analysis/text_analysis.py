import string
import nltk
import random

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from wordcloud import WordCloud, get_single_color_func
from os import path
from collections import Counter
from snlp import logger
from tqdm import tqdm
from nltk.corpus import stopwords
from scipy import stats


import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px


from plotly import tools

# Todo Move this to init 
# nltk.download('stopwords')

def create_adjust_subplots(labels):

    if len(labels) == 0:
        titles = ("Analysis of Text", "Analysis of Labels", 
                 "Document Lengths", "", 
                 "", "", "", "Word Frequency", "", "", "", "", ""
                 "Common Nouns", "", "","", "Common Adjectives","", "","", "Common Verbs")

    elif len(labels) == 1:
        titles = ("Analysis of Text", "Analysis of Labels", 
                 "Document Lengths", labels[0][0].capitalize(), "", 
                 "", "Word Frequency", "", "", "", "",
                 "Common Nouns", "", "", "", "Common Adjectives", "", "", "", "Common Verbs")

    elif len(labels) == 2:
        titles = ("Analysis of Text", "Analysis of Labels", 
                 "Document Lengths", labels[0][0].capitalize(), "", labels[1][0].capitalize(),
                 "Word Frequency", "", "", 
                 "Common Nouns", "", "", "", "Common Adjectives", "", "", "", "Common Verbs")
    
    elif len(labels) == 3:
        titles = ("Analysis of Text", "Analysis of Labels", 
                 "Document Lengths", labels[0][0].capitalize(), "", labels[1][0].capitalize(),
                 "Word Frequency", labels[2][0].capitalize(), "", 
                 "Common Nouns", "", "", "", "Common Adjectives", "", "", "", "Common Verbs")

    elif len(labels) == 4:
        titles = ("Analysis of Text", "Analysis of Labels", 
                 "Document Lengths", labels[0][0].capitalize(), "", labels[1][0].capitalize(),
                 "Word Frequency", labels[2][0].capitalize(), "", 
                 "Common Nouns", labels[3][0].capitalize(), "", "Common Adjectives", "", "", "", "Common Verbs")


    fig = make_subplots(rows=16,
                cols=2,
                subplot_titles=titles,
                specs=[[{}, {}],
                    [{"rowspan": 2}, {"rowspan": 2} if len(labels) >= 1 else {}], # row 2
                    [None, None if len(labels) >= 1 else {}], 
                    [{}, {"rowspan": 2} if len(labels) >= 2 else {}],   # row 4
                    [{"rowspan": 2}, None if len(labels) >= 2 else {}],
                    [None, {"rowspan": 2} if len(labels) >= 3 else {}], # row 6
                    [{}, None if len(labels) >= 3 else {}],
                    [{"rowspan": 3}, {"rowspan": 2} if len(labels) >= 4 else {}], # row 8
                    [None, None if len(labels) >= 4 else {}],
                    [None, {}],
                    [{"rowspan": 3}, {}], # 12
                    [None, {}],
                    [None, {}],
                    [{"rowspan": 3}, {}], # 16
                    [None, {}],
                    [None, {}]
                    ],
                vertical_spacing=0.035)
    return fig


def plotly_wordcloud(token_count_dic):

    wc = WordCloud(color_func=get_single_color_func('deepskyblue'), max_words=100)
    wc.generate_from_frequencies(token_count_dic) 
    
    word_list=[]
    rel_freq_list=[]
    freq_list = []
    fontsize_list=[]
    position_list=[]
    orientation_list=[]
    color_list=[]


    for (word, rel_freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        rel_freq_list.append(rel_freq)
        freq_list.append(token_count_dic[word])
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)
        
    # get the positions
    x=[]
    y=[]
    for i in position_list:
        x.append(i[0])
        y.append(i[1])
            
    # get the relative occurence frequencies
    new_freq_list = []
    for i in rel_freq_list:
        new_freq_list.append(i*100)
    new_freq_list
    
    trace = go.Scatter(x=x, 
                       y=y, 
                       textfont = dict(size=new_freq_list,
                                       color=color_list),
                       hoverinfo='text',
                       hovertext=['{0}: {1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                       mode='text',  
                       text=word_list
                      )
    return trace



def generate_report(df, out_dir, text_col, label_cols=[], language='english', skip_stopwords_punc=True, pos_tags=None):
    """
    Args:
        text_col (string): tokenized text.
        pos_tags (list): Analysis on POS tags. Currently supports NN, JJ, and V
    """
    
    if len(label_cols)>4:
        raise ValueError('Maximum of 4 labels can be specidied for analysis.')

    stop_words = set(stopwords.words(language))
    punctuations = set(string.punctuation)

    doc_lengths = []

    token_to_count = {}
    NNs = {}
    JJs = {}
    Vs = {}

    def update_count(items_dic, items):
        for t in items:
            if t in items_dic:
                items_dic[t] += 1
            else:
                items_dic[t] = 1

    def get_pos(tagged_tokens, pos):
        res = []
        for pt in tagged_tokens:
            if pt[1].startswith(pos):
                res.append(pt[0])
        return res


    def generate_wordcloud(token_count_dic, filename):
        wc = WordCloud(color_func=get_single_color_func('black')).generate_from_frequencies(token_count_dic)
        plt.figure()
        plt.imshow(wc)
        plt.axis("off")
        plt.savefig(path.join(out_dir, filename))

        
    def token2count_to_csv(d, filename):
        with open(filename,'w') as f:
            for k, v in d:
                f.write(str(k)+','+str(v)+'\n')

    
    def generate_plots(doc_length_list, word_freq_list, word_freq_dict, adj_freq_dic, veb_freq_dic):

        # To learn more about customizing plots and x, y labels: https://plotly.com/python/subplots/#customizing-subplot-axes

        # To learn how to make a word cloud with plotly: https://github.com/PrashantSaikia/Wordcloud-in-Plotly/blob/master/plotly_wordcloud.py

        # Leverage Dist Plot
        def distplot_to_dist_trace(data, color):
            data_dist = ff.create_distplot([data], group_labels=['distplot'], colors = [color])['data']
            for item in data_dist:
                item.pop('xaxis', None)
                item.pop('yaxis', None)
            return data_dist 
        

        fig_main = create_adjust_subplots(label_cols)
        
        data1_dist = distplot_to_dist_trace(doc_length_list, color='rgb(0, 200, 200)')
        data2_dist = distplot_to_dist_trace(word_freq_list, color='magenta')
        
        # trace0 = go.Histogram(x=data1, histnorm='probability', name='doc_length')
        
        d1_hist = data1_dist[0]
        d1_kde = data1_dist[1]
        d1_rug = data1_dist[2]

        d2_hist = data2_dist[0]
        d2_kde = data2_dist[1]
        d2_rug = data2_dist[2]

        noun_cloud = plotly_wordcloud(token_count_dic=word_freq_dict)
        adj_cloud = plotly_wordcloud(token_count_dic=adj_freq_dic)
        verb_cloud = plotly_wordcloud(token_count_dic=veb_freq_dic)

        fig_main.append_trace(d1_hist, 2, 1)
        fig_main.append_trace(d1_kde, 2, 1)
        fig_main.append_trace(d1_rug, 4, 1)

        fig_main.append_trace(d2_hist, 5, 1)
        fig_main.append_trace(d2_kde, 5, 1)
        fig_main.append_trace(d2_rug, 7, 1)
        
        fig_main.append_trace(noun_cloud, 8, 1)
        fig_main.append_trace(adj_cloud, 11, 1)
        fig_main.append_trace(verb_cloud, 14, 1)

        # fig_main.update_xaxes(title_text='Length', row=2, col=1)
        fig_main.update_xaxes(rangemode='tozero', row=4, col=1)
        fig_main.update_yaxes(showticklabels=False, row=4, col=1)

        # fig_main.update_xaxes(title_text="Frequency", row=5, col=1)
        fig_main.update_xaxes(rangemode='tozero', row=7, col=1)
        fig_main.update_yaxes(showticklabels=False, row=7, col=1)

        fig_main.update_yaxes(title_text="Probability", row=2, col=1)
        fig_main.update_yaxes(title_text="Probability", row=5, col=1)
        

        fig_main.update_xaxes(showticklabels=False, zeroline=False, row=8, col=1)
        fig_main.update_xaxes(showticklabels=False, zeroline=False, row=11, col=1)
        fig_main.update_xaxes(showticklabels=False, zeroline=False, row=14, col=1)
        fig_main.update_yaxes(showticklabels=False, zeroline=False, row=8, col=1)
        fig_main.update_yaxes(showticklabels=False, zeroline=False, row=11, col=1)
        fig_main.update_yaxes(showticklabels=False, zeroline=False, row=14, col=1)



        # Labels
        def create_label_plot(df, label_col, label_type):
            if label_type == 'categorical':
                values = df[label_col].unique().tolist() # ['pos', 'neg', 'neutral']
                counts = df[label_col].value_counts() # 1212323
                x = []
                y = []
                for v in values:
                    x.append(v)
                    y.append(counts[v])
                trace = go.Bar(x=x, y=y, name=label_col)
            elif label_type == 'numerical':
                trace = go.Histogram(x=df[label_col], name=label_col)
            else:
                raise ValueError('label_col input argument must be set to either \"categorical\" or \"numerical\".')
            return trace

        if len(label_cols) == 1:
            lab_trace1 = create_label_plot(df, label_col=label_cols[0][0], label_type=label_cols[0][1])
            fig_main.append_trace(lab_trace1, 2, 2)
            fig_main.update_yaxes(title_text="Count", row=2, col=2)
        elif len(label_cols) == 2:
            lab_trace1 = create_label_plot(df, label_col=label_cols[0][0], label_type=label_cols[0][1])
            lab_trace2 = create_label_plot(df, label_col=label_cols[1][0], label_type=label_cols[1][1])
            fig_main.append_trace(lab_trace1, 2, 2)
            fig_main.append_trace(lab_trace2, 4, 2)
            fig_main.update_yaxes(title_text="Count", row=2, col=2)
            fig_main.update_yaxes(title_text="Count", row=4, col=2)
        elif len(label_cols) == 3:
            lab_trace1 = create_label_plot(df, label_col=label_cols[0][0], label_type=label_cols[0][1])
            lab_trace2 = create_label_plot(df, label_col=label_cols[1][0], label_type=label_cols[1][1])
            lab_trace3 = create_label_plot(df, label_col=label_cols[2][0], label_type=label_cols[2][1])
            fig_main.append_trace(lab_trace1, 2, 2)
            fig_main.append_trace(lab_trace2, 4, 2)
            fig_main.append_trace(lab_trace3, 6, 2)
            fig_main.update_yaxes(title_text="Count", row=2, col=2)
            fig_main.update_yaxes(title_text="Count", row=4, col=2)
            fig_main.update_yaxes(title_text="Count", row=6, col=2)
        elif len(label_cols) == 4:
            lab_trace1 = create_label_plot(df, label_col=label_cols[0][0], label_type=label_cols[0][1])
            lab_trace2 = create_label_plot(df, label_col=label_cols[1][0], label_type=label_cols[1][1])
            lab_trace3 = create_label_plot(df, label_col=label_cols[2][0], label_type=label_cols[2][1])
            lab_trace4 = create_label_plot(df, label_col=label_cols[3][0], label_type=label_cols[3][1])
            fig_main.append_trace(lab_trace1, 2, 2)
            fig_main.append_trace(lab_trace2, 4, 2)
            fig_main.append_trace(lab_trace3, 6, 2)
            fig_main.append_trace(lab_trace4, 8, 2)
            fig_main.update_yaxes(title_text="Count", row=2, col=2)
            fig_main.update_yaxes(title_text="Count", row=4, col=2)
            fig_main.update_yaxes(title_text="Count", row=6, col=2)
            fig_main.update_yaxes(title_text="Count", row=8, col=2)

       

        
        # Main figure: Update and show
        fig_main.update_layout(height=3100, showlegend=False)
        fig_main.show()
    
        # Add annotation and align 
        # doc_len_description = 'Distribution of document lengths. You can observe what document lengths are most common and which document lengths are too large and based on this observation, you can specify a cut-of threshold'
        # fig_main.layout.annotations[0].update(text=doc_len_description, x=0.1)
        
        


    for text in tqdm(df[text_col]):
        try:
            tokens = text.lower().split(' ')
            doc_lengths.append(len(tokens))
            if skip_stopwords_punc:
                tokens = [t for t in tokens if t not in stop_words and t not in punctuations]
                update_count(token_to_count, tokens)

        except Exception as e:
            logger.warning('Processing entry --- %s --- lead to exception: %s' %(text, e.args[0]))
            continue
        
        if pos_tags:
            postag_tokens = nltk.pos_tag(tokens)
            if 'NN' in pos_tags:
                nouns = get_pos(postag_tokens, 'NN')
                update_count(NNs, nouns)
            if 'VB' in pos_tags:
                verbs = get_pos(postag_tokens, 'VB')
                update_count(Vs, verbs)
            if 'JJ' in pos_tags:
                adjectives = get_pos(postag_tokens, 'JJ')
                update_count(JJs, adjectives)

    sorted_token_to_count = [(k, v) for k, v in sorted(token_to_count.items(), key=lambda item: item[1], reverse=True)]
    token2count_to_csv(sorted_token_to_count, 'token_count.csv')

    word_frequencies = [v for _, v in token_to_count.items()]

    generate_plots(doc_lengths, word_frequencies, NNs, JJs, Vs)

    if pos_tags:
        if 'NN' in pos_tags:
            sorted_nouns_to_count = [(k, v) for k, v in sorted(NNs.items(), key=lambda item: item[1], reverse=True)]
            generate_wordcloud(NNs, 'noun_wordcloud.png')
            token2count_to_csv(sorted_nouns_to_count, 'nount_count.csv')
        if 'JJ' in pos_tags:
            sorted_adjectives_to_count = [(k, v) for k, v in sorted(JJs.items(), key=lambda item: item[1], reverse=True)]
            generate_wordcloud(JJs, 'adjective_wordcloud.png')
            token2count_to_csv(sorted_adjectives_to_count, 'adjectives_count.csv')
        if 'VB' in pos_tags:
            sorted_verbs_to_count = [(k, v) for k, v in sorted(Vs.items(), key=lambda item: item[1], reverse=True)]
            generate_wordcloud(Vs, 'verb_wordcloud.png')
            token2count_to_csv(sorted_verbs_to_count, 'verbs_count.csv')

