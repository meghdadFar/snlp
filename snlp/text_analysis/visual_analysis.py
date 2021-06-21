import string
from typing import List, Tuple
import nltk
import os
import plotly
import pandas
import plotly.figure_factory as ff
import plotly.graph_objs as go

from wordcloud import WordCloud, get_single_color_func
from snlp import logger
from tqdm import tqdm
from nltk.corpus import stopwords
from plotly.subplots import make_subplots


def create_adjust_subplots(labels: List[Tuple]) -> plotly.graph_objects.Figure:
    """Create subplots and adjust the location of the titles wrt the number of labels.

    Args:
        labels (list): List of (label, type) tuples.

    Returns:
        fig (plotly.graph_objs.Figure)
    """
    if len(labels) == 0:
        titles = (
            "Analysis of Text",
            "Analysis of Labels",
            "Document Lengths",
            "",
            "",
            "",
            "",
            "Word Frequency",
            "",
            "",
            "",
            "",
            "" "Common Nouns",
            "",
            "",
            "",
            "Common Adjectives",
            "",
            "",
            "",
            "Common Verbs",
        )
    elif len(labels) == 1:
        titles = (
            "Analysis of Text",
            "Analysis of Labels",
            "Document Lengths",
            labels[0][0].capitalize(),
            "",
            "",
            "Word Frequency",
            "",
            "",
            "",
            "",
            "Common Nouns",
            "",
            "",
            "",
            "Common Adjectives",
            "",
            "",
            "",
            "Common Verbs",
        )
    elif len(labels) == 2:
        titles = (
            "Analysis of Text",
            "Analysis of Labels",
            "Document Lengths",
            labels[0][0].capitalize(),
            "",
            labels[1][0].capitalize(),
            "Word Frequency",
            "",
            "",
            "Common Nouns",
            "",
            "",
            "",
            "Common Adjectives",
            "",
            "",
            "",
            "Common Verbs",
        )
    elif len(labels) == 3:
        titles = (
            "Analysis of Text",
            "Analysis of Labels",
            "Document Lengths",
            labels[0][0].capitalize(),
            "",
            labels[1][0].capitalize(),
            "Word Frequency",
            labels[2][0].capitalize(),
            "",
            "Common Nouns",
            "",
            "",
            "",
            "Common Adjectives",
            "",
            "",
            "",
            "Common Verbs",
        )
    elif len(labels) == 4:
        titles = (
            "Analysis of Text",
            "Analysis of Labels",
            "Document Lengths",
            labels[0][0].capitalize(),
            "",
            labels[1][0].capitalize(),
            "Word Frequency",
            labels[2][0].capitalize(),
            "",
            "Common Nouns",
            labels[3][0].capitalize(),
            "",
            "Common Adjectives",
            "",
            "",
            "",
            "Common Verbs",
        )
    fig = make_subplots(
        rows=16,
        cols=2,
        subplot_titles=titles,
        specs=[
            [{}, {}],
            [{"rowspan": 2}, {"rowspan": 2} if len(labels) >= 1 else {}],  # row 2
            [None, None if len(labels) >= 1 else {}],
            [{}, {"rowspan": 2} if len(labels) >= 2 else {}],  # row 4
            [{"rowspan": 2}, None if len(labels) >= 2 else {}],
            [None, {"rowspan": 2} if len(labels) >= 3 else {}],  # row 6
            [{}, None if len(labels) >= 3 else {}],
            [{"rowspan": 3}, {"rowspan": 2} if len(labels) >= 4 else {}],  # row 8
            [None, None if len(labels) >= 4 else {}],
            [None, {}],
            [{"rowspan": 3}, {}],  # 12
            [None, {}],
            [None, {}],
            [{"rowspan": 3}, {}],  # 16
            [None, {}],
            [None, {}],
        ],
        vertical_spacing=0.035,
    )
    return fig


def plotly_wordcloud(token_count_dic: dict) -> plotly.graph_objects.Scatter:
    """Create a world cloud trace for plotly.

    Args:
        token_count_dic (dictionary): Dictionary of token to its count

    Returns:
        trace (plotly.graph_objects.Scatter)
    """
    wc = WordCloud(color_func=get_single_color_func("deepskyblue"), max_words=100)
    wc.generate_from_frequencies(token_count_dic)

    word_list = []
    rel_freq_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, rel_freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        rel_freq_list.append(rel_freq)
        freq_list.append(token_count_dic[word])
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x = []
    y = []
    for i in position_list:
        x.append(i[0])
        y.append(i[1])

    # get the relative occurence frequencies
    new_freq_list = []
    for i in rel_freq_list:
        new_freq_list.append(i * 100)
    new_freq_list

    trace = go.Scatter(
        x=x,
        y=y,
        textfont=dict(size=new_freq_list, color=color_list),
        hoverinfo="text",
        hovertext=["{0}: {1}".format(w, f) for w, f in zip(word_list, freq_list)],
        mode="text",
        text=word_list,
    )
    return trace


def generate_text_plots(
    figure: plotly.graph_objs.Figure,
    doc_length_list: List,
    word_freq_list: List,
    noun_freq_dict: dict,
    adj_freq_dic: dict,
    verb_freq_dic: dict,
) -> None:
    """Generate distribution plots and word clouds for the textual content based on the input argumnets.

    Args:
        figure (plotly.graph_objs.Figure): Figure object in which the plots are created.
        doc_length_list (list): List containing the length of each document.
        word_freq_list (list): List containing the frequency of each word.
        noun_freq_dict (dictionary): Dictionary of noun to frequency.
        adj_freq_dict (dictionary): Dictionary of noun to frequency.
        verb_freq_dict (dictionary): Dictionary of verb to frequency.

    Returns:
        None
    """
    # Customize plots and x, y labels: https://plotly.com/python/subplots/#customizing-subplot-axes
    # Word cloud with plotly: https://github.com/PrashantSaikia/Wordcloud-in-Plotly/blob/master/plotly_wordcloud.py
    def _distplot_to_dist_trace(data: List, color: str) -> dict:
        """Create a trace from data.
        Args:
            data (list): List of numbers
            color (list[str])

        Returns:
            data_dist (dict): Representation of a distplot figure.
        """
        data_dist = ff.create_distplot([data], group_labels=["distplot"], colors=[color])["data"]
        for item in data_dist:
            item.pop("xaxis", None)
            item.pop("yaxis", None)
        return data_dist

    data1_dist = _distplot_to_dist_trace(doc_length_list, color="rgb(0, 200, 200)")
    data2_dist = _distplot_to_dist_trace(word_freq_list, color="magenta")

    d1_hist = data1_dist[0]
    d1_kde = data1_dist[1]
    d1_rug = data1_dist[2]

    d2_hist = data2_dist[0]
    d2_kde = data2_dist[1]
    d2_rug = data2_dist[2]

    noun_cloud = plotly_wordcloud(token_count_dic=noun_freq_dict)
    adj_cloud = plotly_wordcloud(token_count_dic=adj_freq_dic)
    verb_cloud = plotly_wordcloud(token_count_dic=verb_freq_dic)

    figure.append_trace(d1_hist, 2, 1)
    figure.append_trace(d1_kde, 2, 1)
    figure.append_trace(d1_rug, 4, 1)

    figure.append_trace(d2_hist, 5, 1)
    figure.append_trace(d2_kde, 5, 1)
    figure.append_trace(d2_rug, 7, 1)

    figure.append_trace(noun_cloud, 8, 1)
    figure.append_trace(adj_cloud, 11, 1)
    figure.append_trace(verb_cloud, 14, 1)

    figure.update_xaxes(rangemode="tozero", row=4, col=1)
    figure.update_yaxes(showticklabels=False, row=4, col=1)

    figure.update_xaxes(rangemode="tozero", row=7, col=1)
    figure.update_yaxes(showticklabels=False, row=7, col=1)

    figure.update_yaxes(title_text="Probability", row=2, col=1)
    figure.update_yaxes(title_text="Probability", row=5, col=1)

    figure.update_xaxes(showticklabels=False, zeroline=False, row=8, col=1)
    figure.update_xaxes(showticklabels=False, zeroline=False, row=11, col=1)
    figure.update_xaxes(showticklabels=False, zeroline=False, row=14, col=1)
    figure.update_yaxes(showticklabels=False, zeroline=False, row=8, col=1)
    figure.update_yaxes(showticklabels=False, zeroline=False, row=11, col=1)
    figure.update_yaxes(showticklabels=False, zeroline=False, row=14, col=1)


def generate_label_plots(figure: plotly.graph_objs.Figure, df: pandas.DataFrame, label_cols: str) -> None:
    """Generate histogram and bar plots for the labels in label_cols.

    Args:
        figure (plotly.graph_objs.Figure): Figure object in which the plots are created.
        df (Pandas DataFrame): DataFrame that contains labels specified in label_cols.
        label_cols (list): list of tuples in the form of [('label_1', 'categorical/numerical'),
                           ('label_2', 'categorical/numerical'), ...]

    Returns:
        None
    """
    if len(label_cols) == 1:
        lab_trace1 = label_plot(df, label_col=label_cols[0][0], label_type=label_cols[0][1])
        figure.append_trace(lab_trace1, 2, 2)
        figure.update_yaxes(title_text="Count", row=2, col=2)
    elif len(label_cols) == 2:
        lab_trace1 = label_plot(df, label_col=label_cols[0][0], label_type=label_cols[0][1])
        lab_trace2 = label_plot(df, label_col=label_cols[1][0], label_type=label_cols[1][1])
        figure.append_trace(lab_trace1, 2, 2)
        figure.append_trace(lab_trace2, 4, 2)
        figure.update_yaxes(title_text="Count", row=2, col=2)
        figure.update_yaxes(title_text="Count", row=4, col=2)
    elif len(label_cols) == 3:
        lab_trace1 = label_plot(df, label_col=label_cols[0][0], label_type=label_cols[0][1])
        lab_trace2 = label_plot(df, label_col=label_cols[1][0], label_type=label_cols[1][1])
        lab_trace3 = label_plot(df, label_col=label_cols[2][0], label_type=label_cols[2][1])
        figure.append_trace(lab_trace1, 2, 2)
        figure.append_trace(lab_trace2, 4, 2)
        figure.append_trace(lab_trace3, 6, 2)
        figure.update_yaxes(title_text="Count", row=2, col=2)
        figure.update_yaxes(title_text="Count", row=4, col=2)
        figure.update_yaxes(title_text="Count", row=6, col=2)
    elif len(label_cols) == 4:
        lab_trace1 = label_plot(df, label_col=label_cols[0][0], label_type=label_cols[0][1])
        lab_trace2 = label_plot(df, label_col=label_cols[1][0], label_type=label_cols[1][1])
        lab_trace3 = label_plot(df, label_col=label_cols[2][0], label_type=label_cols[2][1])
        lab_trace4 = label_plot(df, label_col=label_cols[3][0], label_type=label_cols[3][1])
        figure.append_trace(lab_trace1, 2, 2)
        figure.append_trace(lab_trace2, 4, 2)
        figure.append_trace(lab_trace3, 6, 2)
        figure.append_trace(lab_trace4, 8, 2)
        figure.update_yaxes(title_text="Count", row=2, col=2)
        figure.update_yaxes(title_text="Count", row=4, col=2)
        figure.update_yaxes(title_text="Count", row=6, col=2)
        figure.update_yaxes(title_text="Count", row=8, col=2)


def label_plot(df: pandas.DataFrame, label_col: str, label_type: str) -> plotly.graph_objects.Histogram:
    """Create a plot for label_col in df, wrt to label_type.

    Args:
        df (Pandas DataFrame): DataFrame that contains label_col.
        label_col (str): Name of the label column in df that must be plotted.
        label_type (str): Represents the type of label and consequently specifies the type of plot.
                             It can be "numerical" or "categorical".

    Returns:
        trace (plotly.graph_objects.Histogram)
    """
    if label_type == "categorical":
        values = df[label_col].unique().tolist()  # ['pos', 'neg', 'neutral']
        counts = df[label_col].value_counts()  # 1212323
        x = []
        y = []
        for v in values:
            x.append(v)
            y.append(counts[v])
        trace = go.Bar(x=x, y=y, name=label_col)
    elif label_type == "numerical":
        trace = go.Histogram(x=df[label_col], name=label_col)
    else:
        raise ValueError('label_col input argument must be set to either "categorical" or "numerical".')
    return trace


def generate_report(
    df: pandas.DataFrame,
    out_dir: str,
    text_col: str,
    label_cols: List = [],
    language: str = "english",
    skip_stopwords_punc: bool = True,
    save_report: bool = False,
) -> None:
    """Generate analysis report and eitherr renders the report via Plotly show api or saves it offline to html.

    Args:
        df (pandas.DataFrame): DataFrame that contains text and labels.
        out_dir (str): Dir where the report is saved. Required only when save_report option is True.
        text_col (str): Name of the column that contains a tokenized text content.
        label_cols (list): list of tuples in the form of [('label_1', 'categorical/numerical'),
                           ('label_2', 'categorical/numerical'), ...]
        language (str): Language of the text in df[text_col]
        skip_stopwords_punc (bool): Whether or not skip stopwords and punctuations in the analysis. Default: True
        save_report (bool): Whether or not save the report as an html file. Default: False

    Returns:
        None
    """

    def update_count(items_dic: dict, items: List[str]) -> None:
        """Update the corresponding count for each key in  items_dic. w.r.t. terms in items.

        Args:
            items_dic (dict): Dictionary mapping keys to their count
            items (list): List of tokens

        Returns:
            None
        """
        for t in items:
            if t in items_dic:
                items_dic[t] += 1
            else:
                items_dic[t] = 1

    def get_pos(tagged_tokens: List[Tuple(str, str)], goal_pos: str) -> List:
        """Extracts goal_pos POS tags from tagged_tokens.

        Args:
            tagged_tokens (List[Tuple(str, str)]): Contains terms and ther pos tags. E.g.
                                                   [('cat', 'NN'), ('sat', 'VBD'), ('on', 'IN'), ('mat', 'NN')]
            goal_pos (str): Pos tag to look for in tagged_tokens

        Returns:
            res (List(str)): List of tokens with goal_pos pos tag
        """
        res = []
        for pt in tagged_tokens:
            if pt[1].startswith(goal_pos):
                res.append(pt[0])
        return res

    if len(label_cols) > 4:
        raise ValueError("Maximum of 4 labels can be specidied for analysis.")

    stop_words = set(stopwords.words(language))
    punctuations = set(string.punctuation)

    doc_lengths = []
    token_to_count = {}
    NNs = {}
    JJs = {}
    Vs = {}

    logger.info("Processing text in %s column of the input DataFrame..." % text_col)
    for text in tqdm(df[text_col]):
        try:
            tokens = text.lower().split(" ")
            doc_lengths.append(len(tokens))
            if skip_stopwords_punc:
                tokens = [t for t in tokens if t not in stop_words and t not in punctuations]
                update_count(token_to_count, tokens)

        except Exception as e:
            logger.warning("Processing entry --- %s --- lead to exception: %s" % (text, e.args[0]))
            continue

        postag_tokens = nltk.pos_tag(tokens)
        nouns = get_pos(postag_tokens, "NN")
        update_count(NNs, nouns)
        verbs = get_pos(postag_tokens, "VB")
        update_count(Vs, verbs)
        adjectives = get_pos(postag_tokens, "JJ")
        update_count(JJs, adjectives)

    word_frequencies = [v for _, v in token_to_count.items()]

    fig_main = create_adjust_subplots(label_cols)

    logger.info("Generating distplots and word cloud for input text")
    generate_text_plots(fig_main, doc_lengths, word_frequencies, NNs, JJs, Vs)
    logger.info("Generating plots for labels")
    generate_label_plots(fig_main, df, label_cols)
    logger.info("Rendering plots")
    fig_main.update_layout(height=3100, showlegend=False)
    if save_report:
        plotly.offline.plot(fig_main, filename=os.path.join(out_dir, "report.html"))
    else:
        fig_main.show()
