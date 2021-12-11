import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from snlp.
input_text_name = "IMDB Corpus"

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
                                 options=[{'label': 'New York City', 'value': 'NYC'},
                                        {'label': 'Montreal', 'value': 'MTL'},
                                        {'label': 'San Francisco', 'value': 'SF'}], 
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
def show_country(country):
    return html.Div([
                html.H3(f'The Country: {country}'),
                dcc.Graph(
                        figure={
                            'data': [
                                {'x': [1, 2, 3], 'y': [2, 4, 3],
                                'type': 'bar', 'name': 'SF'},
                                {'x': [1, 2, 3], 'y': [5, 4, 3],
                                'type': 'bar', 'name': u'Montréal'}
                                ]
                            }
                        )
            ])

if __name__ == '__main__':
    app.run_server(debug=True)