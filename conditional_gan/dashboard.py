import dash
import dash_daq as daq
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import base64
import torch
from torchvision.utils import save_image
import yaml

from modules import ConditionalGenerator

num_class = 10

# builds a centered row of 'num_class' horizontal sliders
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
    model_file = '../conditional_gan/artifacts/generator.pt'

    # initialize and load model
    generator = ConditionalGenerator(z_dim, num_class, input_dim[-1])
    generator.load_state_dict(torch.load(
        model_file, map_location=torch.device('cpu')))

    # initialize style vector distribution
    style_dist = torch.distributions.normal.Normal(
        torch.zeros(1, z_dim), torch.ones(1, z_dim))

    # sample initial style vector
    style_vec = style_dist.sample()

    # sample initial label vector
    label_vec = torch.zeros((1, num_class))
    label_vec[0, 0] = 1

    # generate sample
    gen_out = generator(style_vec, label_vec)[0]

    # save generated sample to image file
    save_image(gen_out, 'artifacts/dashboard_gen_out.png')

    gen_out_base64 = base64.b64encode(
        open('artifacts/dashboard_gen_out.png', 'rb').read()).decode('ascii')

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
        html.Div([
            html.Img(
                src='data:image/png;base64,{}'.format(gen_out_base64),
                id='gen_out',
                style={
                    'width': '400px',
                    'height': '400px'
                }
            )
        ], style={
            'text-align': 'center',
            'position': 'relative',
            'top': '30px'
        }),
    ])


    @app.callback(
        Output('gen_out', 'src'),
        [Input('{}-slider'.format(str(i)), 'value') for i in range(num_class)])
    def update_output(*values):
        # rescale slider values to probability distribution
        label_vec = torch.tensor(list(values)).unsqueeze(0).float()
        label_vec /= torch.sum(label_vec)

        # generate sample
        gen_out = generator(style_vec, label_vec)[0]

        # save generated sample to image file
        save_image(gen_out, 'artifacts/dashboard_gen_out.png')

        gen_out_base64 = base64.b64encode(
            open('artifacts/dashboard_gen_out.png', 'rb').read()).decode('ascii')

        return 'data:image/png;base64,{}'.format(gen_out_base64)

    app.run_server(debug=True)


if __name__ == '__main__':
    main()
