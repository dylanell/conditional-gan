import yaml
import torch
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_daq as daq
from dash.dependencies import Input, Output

from modules import ConditionalGenerator

num_class = 10

def multi_hot_slider(num_class):
    slider_width = 15
    slider_height = 75
    number_height = 25

    return html.Div(
        [
            html.Div(
                [
                    daq.Slider(
                        id='{}-slider'.format(str(i)),
                        min=0,
                        max=1,
                        step=0.05,
                        value=1*(i == 0),
                        vertical=True,
                        size=75
                    ),
                    html.H6(
                        '{}'.format(i),
                        style={'height': '{}px'.format(number_height)}
                    )
                ],
                style={
                    'position': 'absolute',
                    'margin': 'auto',
                    'top': '0px',
                    'left': '{}px'.format(str(i*(slider_width+20))),
                    'width': '{}px'.format(slider_width),
                    'height': '{}px'.format(slider_height)
                }
            ) for i in range(num_class)
        ],
        style={
            'position': 'relative',
            'margin': 'auto',
            'width': '{}px'.format(str(num_class*(slider_width+19))),
            'height': '{}px'.format(slider_height),
            'text-align': 'center'
        }
    )

def main():
    # parse configuration file
    with open('config.yaml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    num_class = config['number_classes']
    input_dim = config['input_dimensions']
    z_dim = config['z_dimension']

    # model artifact file path
    model_file = 'artifacts/generator.pt'

    # initialize and load model
    generator = ConditionalGenerator(z_dim, num_class, input_dim[-1])
    generator.load_state_dict(
        torch.load(model_file, map_location=torch.device('cpu')))

    # initialize style vector distribution
    style_dist = torch.distributions.normal.Normal(
        torch.zeros(1, z_dim), torch.ones(1, z_dim))

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([
        html.Div(
            html.H2('Conditional GAN Demo'),
            style={'text-align': 'center'}),
        html.Div(html.Br()),
        html.Div(
            html.Button('Sample New Style', id='style_button'),
            style={'text-align': 'center'}),
        html.Br(),
        multi_hot_slider(num_class),
        html.Br(),
        html.Div(id='output')
    ])

    app.run_server(debug=True)

    @app.callback(
    Output('output', 'children'),
    [Input('0-slider', 'value')])
    def update_output(value):
        return 'The slider is currently at {}.'.format(value)


if __name__ == '__main__':
    main()
