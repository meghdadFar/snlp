import types
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from snlp.text_analysis.visual_analysis import generate_report, plotly_wordcloud
import plotly.figure_factory as ff
import plotly.graph_objs as go
from snlp import logger


# Process data
input_text_name = "IMDB Corpus"
imdb_train = pd.read_csv('data/imdb_train_sample.tsv', sep='\t', names=['label', 'text'])
imdb_train = imdb_train.sample(1000)

analysis_res = generate_report(df=imdb_train,
                               out_dir='output_dir',
                               text_col='text',
                               label_cols=[('label', 'categorical')])

dist_setup = {
              'paper_bgcolor': '#007A78',
            }
doc_len_dist = ff.create_distplot([analysis_res.doc_lengths], group_labels=["distplot"], colors=["blue"])
doc_len_dist.update_layout(dist_setup)

fig_w_freq = go.Figure()
fig_w_freq.add_trace(go.Scattergl(x=analysis_res.zipf_x, y=analysis_res.zipf_y_emp, mode='markers'))
fig_w_freq.add_trace(go.Scattergl(x=analysis_res.zipf_x, y=analysis_res.zipf_y_theory, mode='markers'))
fig_w_freq.update_layout(dist_setup)

fig_noun_cloud = go.Figure(plotly_wordcloud(token_count_dic=analysis_res.nns))
fig_verb_cloud = go.Figure(plotly_wordcloud(token_count_dic=analysis_res.vs))
fig_adj_cloud = go.Figure(plotly_wordcloud(token_count_dic=analysis_res.jjs))

word_cloud_setup = {'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                    'xaxis_showgrid':False,
                    'yaxis_showgrid':False,
                    'xaxis_zeroline':False,
                    'yaxis_zeroline':False,
                    'yaxis_visible':False,
                    'yaxis_showticklabels':False,
                    'xaxis_visible':False,
                    'xaxis_showticklabels':False,
                    }



fig_noun_cloud.update_layout(word_cloud_setup)
fig_adj_cloud.update_layout(word_cloud_setup)
fig_verb_cloud.update_layout(word_cloud_setup)

app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])

app.layout = html.Div([
    # First row: Header
    html.Div([
        html.Div([
            html.Img(src=app.get_asset_url('br02-no-bg-white.png'), id='logo', 
            style={'height':'300px', 'width':'auto', 'margin-bottom':'25px'})
        ], className='one-third column'),
        html.Div([
            html.Div([
                html.H3('SNLP', style={'margin-bottom': '0px', 'color': 'white'}),
                html.H5(f'Statistics & Insights from {input_text_name}', style={'margin-bottom': '0px', 'color': 'white'})
            ])
        ], className='one-half column', id = 'title'),
        html.Div([
            html.H6('Last Updated: 00:01 (UTC)',
                    style={'color': 'orange'})

        ], className='one-third column', id = 'title1')

    ], id='header', className='row flex-display', style={'margin-bottom': '25px'}),
    # Second row: Tabs
    html.Div([
        dcc.Tabs(
            id="tabs",
            value='tab-1',  # Default
            parent_className='custom-tabs',
            className='custom-tabs-container',
            children=[
                dcc.Tab(
                    label='Stats',
                    value='tab-1',
                    className='custom-tab',
                    selected_className='custom-tab--selected',
                    style= {'backgroundColor': '#050c22'},
                    selected_style= {'backgroundColor': '#119DFF'},
                ),
                dcc.Tab(
                    label='Part of Speech',
                    value='tab-2',
                    className='custom-tab',
                    selected_className='custom-tab--selected',
                    style= {'backgroundColor': '#050c22'},
                    selected_style= {'backgroundColor': '#119DFF'},
                ),
                dcc.Tab(
                    label='MWEs',
                    value='tab-3',
                    className='custom-tab',
                    selected_className='custom-tab--selected',
                    style= {'backgroundColor': '#050c22'},
                    selected_style= {'backgroundColor': '#119DFF'},
                ),
                dcc.Tab(
                    label='Keyphrases',
                    value='tab-4', 
                    className='custom-tab',
                    selected_className='custom-tab--selected',
                    style= {'backgroundColor': '#050c22'},
                    selected_style= {'backgroundColor': '#119DFF'},
                ),
                dcc.Tab(
                    label='Labels',
                    value='tab-5',
                    className='custom-tab',
                    selected_className='custom-tab--selected',
                    style= {'backgroundColor': '#050c22'},
                    selected_style= {'backgroundColor': '#119DFF'},
                ),
            ]),
        html.Div(id='tabs-content')
    ])
], id = 'mainContainer', style={'display': 'flex', 'flex-direction': 'column'})


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')]
              )
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H4(style={'margin-top': '48px'}),
            # Cards
            html.Div([
                html.Div([
                    html.H6(children='Detected Language/s',
                            style={'textAlign': 'center',
                                'color': 'white'}),
                    html.P(", ".join(analysis_res.languages).strip(),
                            style={'textAlign': 'center',
                                'color': 'orange',
                                'fontSize': 40}),
                    ], className='card_container three columns'),
                html.Div([
                    html.H6(children='Number of Words',
                            style={'textAlign': 'center',
                                'color': 'white'}),
                    html.P(f"Unique Words: {analysis_res.type_count:,}",
                            style={'textAlign': 'center',
                                'color': 'orange',
                                'fontSize': 20}),
                    html.P(f"All Words: {analysis_res.token_count:,}",
                            style={'textAlign': 'center',
                                'color': 'orange',
                                'fontSize': 20}),
                    ], className='card_container three columns'),
                html.Div([
                    html.H6(children='Number of Documents',
                            style={'textAlign': 'center',
                                'color': 'white'}),
                    html.P(f"{analysis_res.doc_count}",
                            style={'textAlign': 'center',
                                'color': 'orange',
                                'fontSize': 40}),
                    ], className='card_container three columns'),
                html.Div([
                    html.H6(children='Median Document Length',
                            style={'textAlign': 'center',
                                'color': 'white'}),
                    html.P(f"{int(analysis_res.median_doc_len)}",
                            style={'textAlign': 'center',
                                'color': 'orange',
                                'fontSize': 40}),
                    ], className='card_container three columns'),
                    
            ], className='row flex display'),
            # Dropdown
            html.Div([
                    html.P('Select Distribution', className = 'fix_label', style = {'color': 'grey', 'margin-top': '2px'}),
                    dcc.Dropdown(id = 'select_option',
                                 multi = False,
                                 clearable = True,
                                 disabled = False,
                                 style = {'display': True},
                                 value = 'Document Length',
                                 placeholder = 'Select Distribution',
                                 options=[{'label': 'Word Frequency', 'value': 'wf'},
                                          {'label': 'Document Length', 'value': 'dl'},
                                         ],
                                        className = 'dcc_compon')],
                                className='dropdown', style={'width': '50%', 'float': 'left'}),
            html.Div(id='dd-content', style={'clear': 'both'}),
        ])
    elif tab == 'tab-2':
        return html.Div([
            # Cards
            html.Div([
                html.Div([
                    html.H6(children='Verbs',
                            style={'textAlign': 'center',
                                'color': 'white'}),
                    html.P(1000,
                            style={'textAlign': 'center',
                                'color': 'orange',
                                'fontSize': 40}),
                    ], className='card_container', style={'width': '16%', 'float': 'left'}),
                html.Div([
                    html.H6(children='Nouns',
                            style={'textAlign': 'center',
                                'color': 'white'}),
                    html.P(f"{10}",
                            style={'textAlign': 'center',
                                'color': 'orange',
                                'fontSize': 40}),
                    ], className='card_container', style={'width': '16%', 'float': 'left'}),
                html.Div([
                    html.H6(children='Adjectives',
                            style={'textAlign': 'center',
                                'color': 'white'}),
                    html.P(f"{200}",
                            style={'textAlign': 'center',
                                'color': 'orange',
                                'fontSize': 40}),
                    ], className='card_container', style={'width': '16%', 'float': 'left'}),
                html.Div([
                    html.H6(children='Adverbs',
                            style={'textAlign': 'center',
                                'color': 'white'}),
                    html.P(f"{300}",
                            style={'textAlign': 'center',
                                'color': 'orange',
                                'fontSize': 40}),
                    ], className='card_container', style={'width': '16%', 'float': 'left'}),
                html.Div([
                    html.H6(children='Named Entities',
                            style={'textAlign': 'center',
                                'color': 'white'}),
                    html.P(f"{300}",
                            style={'textAlign': 'center',
                                'color': 'orange',
                                'fontSize': 40}),
                    ], className='card_container', style={'width': '16%', 'float': 'left'}),
                    
            ], className='row flex display'),
            html.Div([
                    html.H6(f'Nouns'),
                    dcc.Graph(figure=fig_noun_cloud)
                ]),
            html.Div([
                    html.H6(f'Verbs'),
                    dcc.Graph(figure=fig_verb_cloud)
                ]),
            html.Div([
                    html.H6(f'Adjectives'),
                    dcc.Graph(figure=fig_adj_cloud)
                ])
        ])
    elif tab == 'tab-5':
        return html.Div([
            html.H4(children='',
                    style={'textAlign': 'left',
                    'color': 'white'}),
            html.H4(children='Label Analysis from snlp',
                    style={'textAlign': 'left',
                    'color': 'white'}),
            html.H4(children='Label Evaluation from Hawk',
                    style={'textAlign': 'left',
                    'color': 'white'})])

@app.callback(Output('dd-content', 'children'),
              [Input('select_option', 'value')]
              )
def show_country(dist):
    dis_name = 'Word Frequency' if dist=='wf' else ('Document Length' if dist=='dl' else 'Unknown')
    if dist == "wf":
        return html.Div([
                    html.H6(f'Showing the Distribution of: {dis_name}'),
                    dcc.Graph(figure=fig_w_freq)
                ])
    elif dist == "dl":
        return html.Div([
                    html.H6(f'Showing the Distribution of: {dis_name}'),
                    dcc.Graph(figure=doc_len_dist)
                ])

if __name__ == '__main__':
    app.run_server(debug=True)