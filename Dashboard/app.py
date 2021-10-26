# Dash prerequisites
import os
import datetime
import requests
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_trich_components as dtc
import dash_bootstrap_components as dbc
from dash import dash_table
from wordcloud import WordCloud, STOPWORDS
from dash.exceptions import PreventUpdate

# Data handling prerequisite
import pandas as pd
import plotly.express as px
from collections import OrderedDict
import base64
from io import BytesIO
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from nltk.corpus import stopwords

# Extract each pdf page to image
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'NatWest Dashboard'

# ========================= Methods ===========================


def getCurrentYear():
    now = datetime.datetime.now()
    return now.year + 2


def uploaded_files():
    """List the files in the upload directory."""
    files = ["File1.pdf", "File2.pdf"]
    # for filename in os.listdir(UPLOAD_DIRECTORY):
    #   path = os.path.join(UPLOAD_DIRECTORY, filename)
    #    if os.path.isfile(path):
    #      files.append(filename)
    return files


json_lst = []  # for all data
files = uploaded_files()  # For all files dir


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    link = "/insights/" + str(filename)
    return html.A(filename, href=link)


def validateInputs(inputUrl, inputCompany, inputYear):
    pdf_url = str(inputUrl)
    company = str(inputCompany)
    year = str(inputYear)

    message_list = []
    print("Validation: ")
    print(json_lst)
    # Validation 1: Check if link exist in json
    for obj in json_lst:
        print(obj['pdf_url'])
        if(obj['pdf_url'] == pdf_url):
            message_list.append("- Report already exists ")
            break
    # Validation 2: Check if inputCompany exist in json
    for obj in json_lst:
        if(obj['company'] == company and obj['year'] == year):
            message_list.append(
                "- Report with company and year already exist ")
        break
    return message_list


def extract_info(pdf_url, pages, path):
    # check if URL is pdf
    if ".pdf" not in pdf_url:
        print("URL is not a PDF.")
        return "nan"

    try:
        response = requests.get(pdf_url)
    except:
        print("Requests failed.")
        return "nan"

    i = 1
    image_path_obj = {}

    response = requests.get(pdf_url, timeout=30)
    images = convert_from_bytes(response.content)
    print("Extract Info Passed ")
    for i, image in enumerate(images):
        if i not in pages:
            continue
        print(f"==> Convert page {i} of pdf to image...")
        image.save(f"{path}/{str(i)}.png")
        image_path_obj[str(i)] = path + "/" + str(i) + ".png"
    return image_path_obj


def process_pdf_url(inputUrl, inputCompany, inputYear):
    relevant_pages = [1, 2, 3, 4, 5]
    company = str(inputCompany)
    year = str(inputYear)
    pdf_url = str(inputUrl)

    json_obj = {}
    json_obj['company'] = company
    json_obj['year'] = year
    json_obj['pdf_url'] = pdf_url

    json_lst.append(json_obj)
    source = "output/"
    if not os.path.exists(source):  # Check if file exist
        os.mkdir(source)

    path = source + company + '_' + year

    if not os.path.exists(path):
        # file exists
        os.mkdir(path)

    json_obj['images_path'] = extract_info(pdf_url, relevant_pages, path)
    json_lst.append(json_obj)

    return json_lst


# ========================= Sidebar ===========================
icon_db = html.Img(className="icon-nav-link",
                   src=app.get_asset_url('icon-dashboard.png'))
icon_insight = html.Img(className="icon-nav-link",
                        src=app.get_asset_url('icon-insights.png'))
icon_upload = html.Img(className="icon-nav-link",
                       src=app.get_asset_url('icon-upload.png'))
icon_doc = html.Img(className="icon-nav-link",
                    src=app.get_asset_url('icon-doc.png'))
left_content = html.Div(
    className="sidebar",
    children=[
        html.Img(className="logo",
                 src='https://creativereview.imgix.net/content/uploads/2016/10/NW_logo_still_800px.jpg'),
        dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink([icon_db, "Dashboard"], href="/dashboard",
                                        className="nav-Link")),
                dbc.NavItem(dbc.NavLink([icon_insight, "Insights"], href="/insights",
                                        className="nav-Link")),
                dbc.NavItem(dbc.NavLink(
                    [icon_upload, "Upload"], href="/upload", className="nav-Link")),
                dbc.NavItem(dbc.NavLink(
                    [icon_doc, "Documentation"], href="https://github.com/jaokuean/team08-capstone/", className="doc-nav-Link")),
            ],
            vertical=True,
            pills=True,
        ),
    ],
)

# ========================= Main Pages ===========================
# Dashboard Components
dashboard_header = html.Div(
    className="section-header",
    children=[
        "DBS Bank"
    ])
dashboard = html.Div(
    className="section-main",
    children=[
        dashboard_header
    ])

# Insights Components
insights_cards = html.Div(
    # For each image extracted from pipeline, output a card
    # Out: Image, Page(title), Extraction type(desc), keywords(badges)
    [
        dtc.Card(
            className="insights-cards",
            image=app.get_asset_url('PAGE5_IMAGE1_new.jpg'),
            title='Page 35',
            description='Table of Metrics',
            badges=['Carbon Footprint', 'GHG Emissions'],
            style={'display': 'inline-block', "padding": "0.3rem 0.3rem"}
        ),
        dtc.Card(
            image=app.get_asset_url('PAGE5_IMAGE1_new.jpg'),
            title='Page 5',
            description='Table of Metrics',
            badges=['Energy Consumption'],
            style={'display': 'inline-block', "padding": "0.3rem 0.3rem"}
        ),
        dtc.Card(
            image=app.get_asset_url('PAGE5_IMAGE1_new.jpg'),
            title='Page 23',
            description='Table of Metrics',
            badges=['Carbon Footprint', 'Scope 1', 'Scope 2'],
            style={'display': 'inline-block', "padding": "0.3rem 0.3rem"}
        ),
        dtc.Card(
            image=app.get_asset_url('PAGE35_IMAGE1.jpg'),
            title='Page 35',
            description='Table of Metrics',
            badges=['Carbon Footprint', 'GHG Emissions'],
            style={'display': 'inline-block', "padding": "0.3rem 0.3rem"}
        ),
        dtc.Card(
            image=app.get_asset_url('PAGE35_IMAGE1.jpg'),
            title='Page 35',
            description='Table of Metrics',
            badges=['Carbon Footprint', 'GHG Emissions'],
            style={"padding": "0.3rem 0.3rem", 'display': 'inline-block'}
        ),
        dtc.Card(
            image=app.get_asset_url('PAGE35_IMAGE1.jpg'),
            title='Page 35',
            description='Table of Metrics',
            badges=['Carbon Footprint', 'GHG Emissions'],
            style={"padding": "0.3rem 0.3rem", 'display': 'inline-block'}
        )
    ]
)
insights_header = html.Div(
    className="section-header",
    children=[
        "DBS Bank"
    ])
insights = html.Div(
    className="section-main",
    children=[
        insights_header,
        insights_cards,
        html.Img(className="logo",
                 src='output/UOB_2020/1.png'),

    ])

# Upload Components
upload_header = html.Div(
    className="section-header",
    children=[
        html.H4("Upload Files")
    ])
upload_content = html.Div(className="upload-main-container", children=[
    html.Div(className="upload-sidebar", children=[
        html.H4("Directory"),
        html.Ul(id="file-list"),
    ]),
    html.Div(className="upload-right-container", children=[
        html.H4(className="upload-sub-header", children=["Upload a PDF file"]),
        dcc.Upload(
            id="upload-pdf",
            className="file-selection-box",
            children=[
                'Drag and Drop or ',
                html.A('Select a File')
            ]),
        html.H4(className="upload-sub-divider",
                children=["OR"], style={'text-align': 'center'}),
        html.H4(className="upload-sub-header",
                children=["Using URL of PDF report"]),
        html.H5(className="upload-sub-header",
                children=[" URL link (.pdf)"]),
        dcc.Input(
            id="inputUrl".format("url"),
            className="upload-right-inputs",
            type='url',
            placeholder="Paste URL link here", size='100',
        ),
        html.Div(id="upload-url-output"),
        html.H5(className="upload-sub-header",
                children=["Company Name"]),
        dcc.Input(
            id="inputCompany".format("text"),
            className="upload-right-inputs",
            type='text',
            placeholder="e.g. DBS", size='45',
        ),
        html.Div(id="upload-companyname-output"),
        html.H5(className="upload-sub-header",
                children=["Year of Report"]),
        dcc.Dropdown(
            id="inputYear",
            placeholder="Select Year",
            options=[{'label': x, 'value': x}
                     for x in reversed(range(1999, getCurrentYear()))],
            style={
                'border': "none",
                'margin-top': '10px',
                'border-bottom': '1px solid #adadad'
            }
        ),
        html.Div(id="upload-year-output"),
        html.Button(className="upload-right-button", children=[
            'Submit'], id='submit-val', n_clicks=0),

        html.Div(id="upload-output"),
    ]),

])
upload_fn = html.Div(
    className="section-main",
    children=[
        upload_header,
        upload_content
    ])

right_content = html.Div(
    id="page-content"
)

index = html.Div(
    dcc.Link('dashboard', refresh=True, href='/dashboard')
)

app.layout = html.Div(className='content', children=[
    dcc.Location(id="url", refresh=True), left_content, right_content])

# Ignore line of code below V
# html.Div(className="syle",children=[])

# ========================= Back end callbacks ===========================
# Callback for NavBar


@ app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return index
    elif pathname == "/dashboard":
        # TODO: Pass filename into dashboard
        return dashboard
    elif "/insights" in pathname:
        # TODO: Pass filename into insights
        return insights
    elif pathname == "/upload":
        return upload_fn
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            # html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


# Callback for uploading files submission

# Validation for URL link
# Test Link:
# https://www.uobgroup.com/AR2020/documents/UOB-Sustainability-Report-2020.pdf
@ app.callback(
    Output("upload-url-output", "children"),
    Input("inputUrl", "value"))
def url_render(inputUrl):
    if (inputUrl is None):
        raise PreventUpdate
    else:
        pass
    # Not Valid: if url does not exist in Json file
    if("http" not in inputUrl):
        return u'URL is not a valid link'
    try:
        response = requests.get(inputUrl)
        # ExtractPDF function -> check if link can extract pdf
    except:
        return u"URL link cannot be reached. "
    else:  # valid case
        return u"Valid URL link "

# Validation for Company Name


@ app.callback(
    Output("upload-companyname-output", "children"),
    Input("inputCompany", "value"))
def companyName_render(inputCompany):
    if (inputCompany is None):
        raise PreventUpdate
    else:
        pass
    if(len(inputCompany) < 2):
        return u'Please enter a valid company name, e.g. DBS'
    else:  # valid case
        return

# Validation for Years


@app.callback(
    Output('upload-year-output', 'children'),
    [Input("inputYear-val", "value")])
def update_output(value):
    if (value is None):
        raise PreventUpdate
    else:
        pass
    if value is None:
        return "Please select a year."
    else:
        return None


# Validation upon clicking submit btn


@ app.callback(
    Output("upload-output", "children"),
    Input('submit-val', 'n_clicks'),
    [
        State("inputUrl", "value"),
        State("inputCompany", "value"),
        State("inputYear", "value")
    ],
)
def input_render(n_clicks, inputUrl, inputCompany, inputYear):
    if (n_clicks is None) or (inputUrl is None) or (inputCompany is None) or (inputYear is None):
        raise PreventUpdate
    else:
        pass
    print("Validation started: ")
    err_message_list = validateInputs(inputUrl, inputCompany, inputYear)
    print("Validation complete: ")
    if(err_message_list == []):
        output_json = process_pdf_url(inputUrl, inputCompany, inputYear)
        files.append(inputCompany)
        print(json_lst)
        return dcc.Location(pathname="/insights", id="url")
    else:
        return [html.Ol(className="err-list-ol", children=[err]) for err in err_message_list]
    # TODO: Process into input file and append json
    # TODO: Run info extraction pipeline and append json
    # TODO: Update file list
    # TODO-extra: Loading view to process file and estimated time left

# Updating file list


@app.callback(
    Output("file-list", "children"),
    Input('submit-val', 'n_clicks'),
    State("inputCompany", "value"),
)
def update_output(n_clicks, inputCompany):
    # files.append(inputCompany)
    return [html.Ol(className="file-list-li", children=[file_download_link(filename)]) for filename in files]


if __name__ == "__main__":
    app.run_server(debug=True)
