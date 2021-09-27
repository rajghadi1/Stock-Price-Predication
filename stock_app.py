import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import dash_bootstrap_components as dbc



# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

VALID_USERNAME_PASSWORD_PAIRS = {
    'tushar': 'tushar'
}
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)


scaler = MinMaxScaler(feature_range=(0, 1))

df_nse = pd.read_csv("NSE-TATA.csv")

df_nse["Date"] = pd.to_datetime(df_nse.Date, format="%Y-%m-%d")
df_nse.index = df_nse['Date']

data = df_nse.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data["Date"][i] = data['Date'][i]
    new_data["Close"][i] = data["Close"][i]

new_data.index = new_data.Date
new_data.drop("Date", axis=1, inplace=True)

dataset = new_data.values

train = dataset[0:987, :]
valid = dataset[987:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []

for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = load_model("saved_model.h5")

inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price

df = pd.read_csv("stock_data/stock_data.csv")

bottom_card1 = dbc.Card(
    [
        dbc.CardBody(html.P("Tushar Raut", className="card-text")),
        dbc.CardImg(src=app.get_asset_url('pic4.jfif'), bottom=True),
    ],
    style={"width": "18rem"},
)
bottom_card2 = dbc.Card(
    [
        dbc.CardBody(html.P("Omkar Samal", className="card-text")),
        dbc.CardImg(src=app.get_asset_url('omkar1.jpg'), bottom=True),
    ],
    style={"width": "18rem"},
)
bottom_card3 = dbc.Card(
    [
        dbc.CardBody(html.P("Mukesh Chawda", className="card-text")),
        dbc.CardImg(src=app.get_asset_url('mukesh1.jpg'), bottom=True),
    ],
    style={"width": "18rem"},
)
cards = dbc.Row(
    [
        dbc.Col(bottom_card1, width="auto"),
        dbc.Col(bottom_card2, width="auto"),
        dbc.Col(bottom_card3, width="auto"),
    ]
)

app.layout = html.Div([

    html.H2("Stock Price Analysis Dashboard", style={"textAlign": "center","padding-bottom":"2%","padding-top":"2%"}),

    dcc.Tabs(id="tabs",  children=[

        dcc.Tab(label="Home", children=[
            #,style={"backgroundColor":"#000000",'color': 'white'}
          html.Div([
html.Br(),
            html.Br(),
          html.Div([
               html.Img(src=app.get_asset_url('front1.png'), style={"padding-left":"4%","padding-right":"4%","height":"70%","width":"100%"})
          ]),
            html.Br(),
          html.Div([
              html.H3("K J Somaiya Institute of Engineering and Information Technology", style={"textAlign": "center"})
          ]),

            html.Br(),
            html.Div([
               html.Img(src=app.get_asset_url('download2.png'),style={"padding-left":"37%"})
          ]),
            html.Br(),
          html.Div([


               html.H4("Group Members", style={"padding-left":"44%"}),
              html.Br(),
               # html.P("Tushar Raut", style={"padding-left":"47%"}),
               # html.P("Omkar Samal", style={"padding-left":"47%"}),
               # html.P("Mukesh Chawda", style={"padding-left":"47%"}),
                html.Div(dbc.Row([cards]), style={"padding-left":"20%"}),
              html.Br(),
               html.H4("Project Guide", style={"padding-left":"44%"}),
               html.P("Ganesh Wadmare", style={"padding-left":"45%"}),


          ]),

          ],style={"backgroundColor":"#F0F3F4",'color': 'black'})





        ]
        ),

        dcc.Tab(label='NSE-TATA GLOBAL Stock Data', children=[
            html.Div([
                html.H2("Actual closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=train.index,
                                y=valid["Close"],
                                mode='lines'
                            )

                        ],
                        "layout": go.Layout(
                            title='Line plot',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                ),
                html.H2("LSTM Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid.index,
                                y=valid["Predictions"],
                                mode='lines'
                            )

                        ],
                        "layout": go.Layout(
                            title='Line plot',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                )
            ])

        ]),
        dcc.Tab(label='US Stock Data', children=[
            html.Div([
                html.H2("US Stocks High vs Lows",
                        style={'textAlign': 'center'}),

                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple', 'value': 'AAPL'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft', 'value': 'MSFT'}],
                             multi=True, value=['FB'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-None": "auto", "width": "100%"}),
                dcc.Graph(id='highlow'),
                html.H2("Stock Market Volume", style={'textAlign': 'center'}),

                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple', 'value': 'AAPL'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft', 'value': 'MSFT'}],
                             multi=True, value=['FB'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "100%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])

    ])
]

)



@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft", }
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
            go.Scatter(x=df[df["Stock"] == stock]["Date"],
                       y=df[df["Stock"] == stock]["High"],
                       mode='lines', opacity=0.7,
                       name=f'High {dropdown[stock]}', textposition='bottom center'))
        trace2.append(
            go.Scatter(x=df[df["Stock"] == stock]["Date"],
                       y=df[df["Stock"] == stock]["Low"],
                       mode='lines', opacity=0.6,
                       name=f'Low {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
                                  xaxis={"title": "Date",
                                         'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'count': 6, 'label': '6M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'step': 'all'}])},
                                         'rangeslider': {'visible': True}, 'type': 'date'},
                                  yaxis={"title": "Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft", }
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
            go.Scatter(x=df[df["Stock"] == stock]["Date"],
                       y=df[df["Stock"] == stock]["Volume"],
                       mode='lines', opacity=0.7,
                       name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
                                  xaxis={"title": "Date",
                                         'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'count': 6, 'label': '6M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'step': 'all'}])},
                                         'rangeslider': {'visible': True}, 'type': 'date'},
                                  yaxis={"title": "Transactions Volume"})}
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)