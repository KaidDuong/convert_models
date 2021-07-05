import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import datetime
from flask_caching import Cache
import os
import pandas as pd
import time
import uuid
import numpy as np
import plotly.graph_objects as go
import dash_table
import base64
import io

def create_layout(app):
    app.layout = html.Div([dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
        ),
        dcc.Tabs(id="tabs-styled-with-props", value='tab-1', children=[
            dcc.Tab(label='Graph', value='tab-1'),
            dcc.Tab(label='Table', value='tab-2'),
        ], colors={
            "border": "white",
            "primary": "gold",
            "background": "cornsilk"
        }),
        html.Div(id='tabs-content-props'),
        # dcc.Store inside the app that stores the intermediate value
        dcc.Store(id='intermediate-value')
    ])
    return app

def draw_para_graph(df):
    @app.callback(Output('slider-drag-output', 'children'),
                  Input('input-slider', 'value'))
    def display_value(value):
        return 'Frame/ Image ID: {}'.format(value)

    @app.callback(
        Output('graph-with-slider', 'figure'),
        Input('input-slider', 'value'),
        Input('range-slider', 'value'))
    def update_figure(selected_frame, select_cols):
        filtered_df = df[df.frame_idxs == selected_frame]
        columns = [f"feature_{i}" for i in range(select_cols[0], select_cols[1])]
        columns.append("model_id")
        # labels = {k: v for k, v in zip(columns, ['Drawings', 'Hentai', 'Neutral', 'Porn', 'Sexy', 'Model IDs'])}
        fig = px.parallel_coordinates(filtered_df, color='model_id',
                                      dimensions=columns,
                                      color_continuous_scale=px.colors.diverging.Armyrose,
                                      color_continuous_midpoint=1)
        fig.update_layout(
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )
        fig.update_layout(transition_duration=500)

        return fig

    @app.callback(
        Output('input-slider', 'value'),
        Input('graph_distance', 'hoverData'))
    def update_slider(hoverData):
        if hoverData is not None:
            return hoverData["points"][0]["x"]
        else:
            return df['frame_idxs'].min()




def store_data(app):
    @app.callback(Output('intermediate-value', 'data'),
                  Input('upload-data', 'contents'),
                  State('upload-data', 'filename'))
    def clean_data(contents, filename):
        if contents is not None:
            # some expensive clean data step
            cleaned_df = parse_contents(contents[0], filename[0])
            #draw_para_graph(cleaned_df)
            return cleaned_df.to_json(date_format='iso', orient='split')
    return app
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            return  df
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            return df
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])



def draw_scater_plot(df):
    columns = [f"feature_{i}" for i in range(255)]
    columns.append("model_id")
    dff = df[columns]
    df2 = pd.DataFrame(np.sqrt(np.sum(
        np.square(dff[dff.model_id == 0].iloc[:, :-1].values[:-1] - dff[dff.model_id == 1].iloc[:, :-1].values[:-1]),
        axis=1)), columns=["l2_distance"])
    df3 = pd.DataFrame(np.sqrt(np.sum(
        np.square(dff[dff.model_id == 0].iloc[:, :-1].values[:-1] - dff[dff.model_id == 2].iloc[:, :-1].values[:-1]),
        axis=1)), columns=["l2_distance"])
    df4 = pd.DataFrame(np.sqrt(np.sum(
        np.square(dff[dff.model_id == 1].iloc[:, :-1].values[:-1] - dff[dff.model_id == 2].iloc[:, :-1].values[:-1]),
        axis=1)), columns=["l2_distance"])

    fig3 = go.Figure()
    # Add traces
    fig3.add_trace(go.Scatter(x=df2.index.values, y=df2.l2_distance.values,
                              name='Onnx vs PTH', mode="markers"))
    fig3.add_trace(go.Scatter(x=df3.index.values, y=df3.l2_distance.values,
                              name='TensorRT vs PTH', mode="markers"))
    fig3.add_trace(go.Scatter(x=df4.index.values, y=df3.l2_distance.values,
                              name='Onnx vs TensorRT', mode="markers"))
    fig3.update_layout(
        title={
            'text': "The difference between the other Models ",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Frame/Image Idxes",
        yaxis_title="Euclidean distance",
        legend_title="Models",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    fig3.update_yaxes(type="log", range=[-2, 0.2])
    return fig3
def create_table(df):
    return html.Div([
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line
    ])



def create_tab(app):
    @app.callback(Output('tabs-content-props', 'children'),
                  Input('tabs-styled-with-props', 'value'),
                  Input('intermediate-value', 'data')
                  )
    def render_content(tab, json_path):
        if json_path is not None:
            df = pd.read_json(json_path, orient='split')
            if tab == 'tab-1':
                fig = draw_scater_plot(df)
                draw_para_graph(df)
                return [
                dcc.Graph(
                    id='graph_distance',
                    figure=fig
                ),
                html.Div(id='label-graph-slider', children='Type of Models : 0 - PTH , 1 - Onnx, 2 - TensorRT',
                         style={'marginTop': 5, 'marginLeft': 1500}),
                dcc.Graph(id='graph-with-slider'),
                html.Div(id='slider-drag-output', style={'marginTop': 5, 'marginLeft': 20}),
                dcc.Slider(
                    id='input-slider',
                    min=df['frame_idxs'].min(),
                    max=df['frame_idxs'].max(),
                    # value=df['frame_idxs'].min(),
                    # marks={str(idx): str(idx) for idx in df['frame_idxs'].unique()[:100]},
                    step=1
                ),
                html.Div(id='label-rslider', children='Columns:', style={'marginTop': 5, 'marginLeft': 20}),
                dcc.RangeSlider(
                    id='range-slider',
                    min=0,
                    max=256,
                    step=1,
                    value=[5, 15],
                    marks={
                        0: {'label': '0', 'style': {'color': '#77b0b1'}},
                        50: {'label': '50'},
                        100: {'label': '100'},
                        200: {'label': '200'},
                        255: {'label': '255', 'style': {'color': '#f50'}}
                    }
                )
            ]
            elif tab == 'tab-2':
                return create_table(df)

    return app


external_stylesheets = [
    # Dash CSS
    'https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True
app = create_layout(app)
app = store_data(app)
app = create_tab(app)
if __name__ == '__main__':

    app.run_server(debug=True)