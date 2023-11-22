"""Dash app to show calculation of Representative Elementary Volumes

(c) CG3, Florian Wellmann, 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import dash
from dash import html, dcc, Input, Output, State
import base64
# import io
from PIL import Image
import plotly.express as px

import plotly.graph_objs as go

from io import BytesIO

class analyse_REV(object):
    """Class for REV estimation"""

    def __init__(self) -> None:
        """Class for REV estimation"""
        self.rev_img = None 
        self.rev_img_gray = None
        self.rev_img_bw = None

    def load_picture(self, filename) -> None:
        """Load picture to analyse REV and convert to grayscale"""
        self.rev_img = mpimg.imread(filename)

        # Convert to grayscale
        r, g, b = cp[:,:,0], cp[:,:,1], cp[:,:,2]
        self.rev_img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    def set_pic_threshold(self, thres=150.) -> None:
        """Set threshold for picture and save result, e.g.: create a b/w 0/1 version for REV analysis"""
        self.rev_bw_img[self.rev_img_gray > thres] = 1
        self.rev_bw_img[self.rev_img_gray < thres] = 0

    def calculate_rev(self, start_point) -> None:
        """Calculate rev for start point (default: center)"""
        pass

def image_to_rgb_array(image_content):
    if image_content is not None:
        decoded_image = parse_contents(image_content)
        # Open the image using PIL
        image = Image.open(BytesIO(decoded_image))
        # Convert the image to RGB format if not already
        image = image.convert('RGB')
        # Convert the PIL image to a NumPy array
        rgb_array = np.array(image)

        return rgb_array
    


def gamma_val(bw_img, x_0, y_0, extent):
    """Determine gamma value from image for given point and extent
    **Arguments**:
        - *img* = 2D ndarray
        - *x_0, y_0* = int, int : central point for REV
        - *extent* = int : extent of REV
    
    **Returns**:
        char_val = float : value of characteristic function
    """
    # extract sub-area from image
    bw_sub = bw_img[x_0-extent//2:x_0+extent//2,
               y_0-extent//2:y_0+extent//2]
    # determine gamma value (note: simply the mean value)
    char_val = bw_sub.mean()

    return char_val


# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    # Row containing both the upload component and the image display
    html.Div([
        # Column for the image upload
        html.Div([
            dcc.Upload(
                id='upload-image',
                children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                style={
                    'width': '100%', 'height': '300px', 'lineHeight': '300px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px'
                },
                multiple=True
            )
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 20px'}),  # Adjust width as needed

        # Column for the image display
        html.Div([
            html.Div(id='output-image-upload')
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 20px'}),  # Adjust width as needed
    ], style={'display': 'flex', 'flex-direction': 'row'}),

    # Slider for threshold input
    html.Div([
        dcc.Slider(
            id='threshold-slider',
            min=0,  # Minimum value of the slider
            max=255,  # Maximum value of the slider
            value=150,  # Initial value of the slider
            marks={i: str(i) for i in range(0, 256, 50)},  # Marks on the slider
            step=1  # Step size of the slider
        )
    ], style={'margin': '20px'}),

    # Slider for px (center point x-coordinate)
    dcc.Slider(
        id='px-slider',
        min=0,
        max=2000,  # Adjust max value based on your image dimensions
        value=1000,  # Default or initial value
        marks={i: str(i) for i in range(0, 2001, 100)},
        step=1,
        tooltip={"placement": "bottom", "always_visible": True}
    ),

    # Slider for py (center point y-coordinate)
    dcc.Slider(
        id='py-slider',
        min=0,
        max=2000,  # Adjust max value based on your image dimensions
        value=1000,  # Default or initial value
        marks={i: str(i) for i in range(0, 2001, 100)},
        step=1,
        tooltip={"placement": "bottom", "always_visible": True}
    ),

    # dcc.Input(id='threshold-input', type='number', placeholder='Enter threshold value'),
    html.Button('Run Analysis', id='analysis-button'),

    # Second row with the processed image and the analysis plot
    html.Div([
        # Column for the processed image
        html.Div([
            dcc.Graph(id='processed-image-display')  # You'll update this in your callback
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 20px'}),

        # Column for the analysis plot
        html.Div([
            dcc.Graph(id='output-graph')  # This will display the analysis results
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 20px'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),

])




@app.callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents')
)
def update_output(list_of_contents):
    if list_of_contents is not None:
        children = [
            parse_contents_show(c) for c in list_of_contents
        ]
        return children

def parse_contents_show(contents):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    return html.Img(src='data:image/png;base64,' + base64.b64encode(decoded).decode(),
                    style={'width': '100%', 'height': 'auto', 'max-height': '300px'})

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return decoded


@app.callback(
    Output('processed-image-display', 'figure'),
    [Input('threshold-slider', 'value'),
     Input('px-slider', 'value'),
     Input('py-slider', 'value')],
    [State('upload-image', 'contents')]
)
def update_image_and_analysis(threshold, px, py, image_content):
    if image_content is None:
        raise dash.exceptions.PreventUpdate

    if image_content[0] is not None:
        # Decode the image content
        decoded_image = parse_contents(image_content[0])
        image = Image.open(BytesIO(decoded_image))

        rgb_image = image.convert('RGB')
        # Convert PIL Image to NumPy array
        rgb_image = np.array(rgb_image)

        # Convert to grayscale
        r, g, b = rgb_image[:,:,0], rgb_image[:,:,1], rgb_image[:,:,2]
        rev_img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        # Apply threshold
        rev_bw_img = np.where(rev_img_gray > threshold, 255, 0)  # Use the threshold from the slider

        # Create Plotly figure
        fig = go.Figure()

        # Add the image data to the figure with a binary colorscale
        fig.add_trace(go.Heatmap(z=rev_bw_img, colorscale=[[0, 'black'], [1, 'white']], showscale=False))

        for d in np.arange(200,1001,200):
            add_rectangle(fig, px, py, d)


        # Optionally, add a marker for the center point
        fig.add_trace(go.Scatter(x=[px], y=[py], mode='markers', marker=dict(color='Blue', size=10)))


        # Update layout so that the image fills the space
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, t=0, b=0),
            # plot_bgcolor='rgba(0,0,0,0)'
        )

        # Update all annotations to have a larger font size
        fig.update_annotations(font_size=20)

        return fig.to_dict()


def add_rectangle(fig, px, py, d):
    """Add rectangle to figure with size d around point px,py"""
    # Calculate square coordinates based on px, py, and d
    x0, y0 = px - d/2, py - d/2  # Bottom left corner
    x1, y1 = px + d/2, py + d/2  # Top right corner

    # Add a square shape
    fig.add_shape(
        type="rect",
        x0=x0, y0=y0, x1=x1, y1=y1,
        line=dict(color="Red"),
    )



@app.callback(
    Output('output-graph', 'figure'),
    [Input('analysis-button', 'n_clicks')],
    [State('upload-image', 'contents'),
     State('threshold-slider', 'value'),
     State('px-slider', 'value'),
     State('py-slider', 'value')]
)
def update_graph(n_clicks, image_content, thres, px, py):
    if n_clicks is None or image_content is None:
        raise dash.exceptions.PreventUpdate

    # Perform analysis on the image
    figure = perform_analysis(thres, px, py, image_content[0])
    return figure

def perform_analysis(thres, px, py, image_content):
    if image_content is not None:
        # Decode the image content
        decoded_image = parse_contents(image_content)
        image = Image.open(BytesIO(decoded_image))

        rgb_image = image.convert('RGB')
        # Convert PIL Image to NumPy array
        rgb_image = np.array(rgb_image)

        # Convert to grayscale
        r, g, b = rgb_image[:,:,0], rgb_image[:,:,1], rgb_image[:,:,2]
        rev_img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
        # apply threshold
        rev_bw_img = np.where(rev_img_gray > thres, 255, 0)  # Use the threshold from the slider

        # calc         
        # define central point
        # x_0, y_0 = 1200, 1600
        # define max extent of REV 
        max_extent = 1000

        # now: determine entire extent range
        char_vals = []
        for extent in range(2, max_extent):
            char_vals.append(gamma_val(rev_bw_img, px, py, extent))

        # create plot
        # Create a Plotly figure
        fig = go.Figure()
        # Add a line plot
        fig.add_trace(go.Scatter(y=char_vals, mode='lines', name='Line Plot'))

        
        for d in np.arange(200,1001,200):
            fig.add_shape(
                type="line",
                yref="paper",  # Reference the entire plot for the y position
                x0=d, y0=0,  # Start point of the line
                x1=d, y1=1,  # End point of the line
                line=dict(color="Red", width=2)
            )


        return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)












