import types
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from snlp.text_analysis.visual_analysis import generate_report
import plotly.figure_factory as ff


# Process data
input_text_name = "IMDB Corpus"
imdb_train = pd.read_csv('../../data/imdb_train_sample.tsv', sep='\t', names=['label', 'text'])
# imdb_train = imdb_train.sample(10)
doc_lengths, word_frequencies = generate_report(df=imdb_train,
                                    out_dir='output_dir',
                                    text_col='text',
                                    label_cols=[('label', 'categorical')])

word_freq_dist = ff.create_distplot([word_frequencies], group_labels=["distplot"], colors=["magenta"], curve_type='normal')
doc_len_dist = ff.create_distplot([doc_lengths], group_labels=["distplot"], colors=["blue"])
word_freq_dist_px = px.histogram(x=word_frequencies, marginal="rug", nbins=50)

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
            value='tab-2',
            parent_className='custom-tabs',
            className='custom-tabs-container',
            children=[
                dcc.Tab(
                    label='Text - Stats',
                    value='tab-1',
                    className='custom-tab',
                    selected_className='custom-tab--selected',
                    style= {'backgroundColor': '#050c22'},
                    selected_style= {'backgroundColor': '#119DFF'},
                ),
                dcc.Tab(
                    label='Text - Extracts ',
                    value='tab-2',
                    className='custom-tab',
                    selected_className='custom-tab--selected',
                    style= {'backgroundColor': '#050c22'},
                    selected_style= {'backgroundColor': '#119DFF'},
                ),
                dcc.Tab(
                    label='Labels',
                    value='tab-3', 
                    className='custom-tab',
                    selected_className='custom-tab--selected',
                    style= {'backgroundColor': '#050c22'},
                    selected_style= {'backgroundColor': '#119DFF'},
                ),
                dcc.Tab(
                    label='Insights',
                    value='tab-4',
                    className='custom-tab',
                    selected_className='custom-tab--selected',
                    style= {'backgroundColor': '#050c22'},
                    selected_style= {'backgroundColor': '#119DFF'},
                ),
            ]),
        html.Div(id='tabs-content')
    ])
], id = 'mainContainer', style={'display': 'flex', 'flex-direction': 'column'})


# @app.callback(Output('tabs-content', 'children'),
#               [Input('tabs', 'value'),
#               Input('tabs', 'value')]
#               )
@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')]
              )
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Tab content 1'),
            html.Div([
                    html.P('Select Country', className = 'fix_label', style = {'color': 'black', 'margin-top': '2px'}),
                    dcc.Dropdown(id = 'select_option',
                                 multi = False,
                                 clearable = True,
                                 disabled = False,
                                 style = {'display': True},
                                 value = 'Switzerland',
                                 placeholder = 'Select Countries',
                                 options=[{'label': 'Word Frequency', 'value': 'wf'},
                                        {'label': 'Documen Length', 'value': 'dl'},
                                        {'label': 'Px', 'value': 'px'}
                                        ], 
                                        className = 'dcc_compon')
            ]),
            html.Div(id='dd-content')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab content 2')
        ])

@app.callback(Output('dd-content', 'children'),
              [Input('select_option', 'value')]
              )
def show_country(dist):
    if dist == "wf":
        return html.Div([
                    html.H3(f'Showing the Distribution of: {dist}'),
                    dcc.Graph(figure=word_freq_dist)
                ])
    elif dist == "dl":
        return html.Div([
                    html.H3(f'Showing the Distribution of: {dist}'),
                    dcc.Graph(figure=doc_len_dist)
                ])
    elif dist == "px":
        return html.Div([
                    html.H3(f'Showing the Distribution of: {dist}'),
                    dcc.Graph(figure=word_freq_dist_px)
                ])

if __name__ == '__main__':
    app.run_server(debug=True)