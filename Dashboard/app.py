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
from flask import request
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
json_input = [
    {
        "company": "ASAHI",
        "year": "2020",
        "pdf_url": "https://www.asahi-life.co.jp/english/annual_report/AnnualReport2020.pdf ",
        "wordcloud_img_path": "wordcloud_images/ASAHI_2020.png",
        "table_images": {1: ['output/ASAHI_2020/1.png', 'output/ASAHI_2020/2.png'], 2:['output/ASAHI_2020/2.png']},
        "table_keywords": {},
        "chart_images": {1: ['output/ASAHI_2020/3.png', 'output/ASAHI_2020/4.png'], 2:['output/ASAHI_2020/5.png']},
        "sentiment_score": {},
        "text_output": {}
    },
    {
        "company": "UOB",
        "year": "2020",
        "pdf_url": "https://www.uobgroup.com/AR2020/documents/UOB-Sustainability-Report-2020.pdf",
        "wordcloud_img_path": "wordcloud_images/UOB_2020.png",
        "table_images": {1: ['output/UOB_2020/1.png', 'output/UOB_2020/2.png'], 2:['output/UOB_2020/2.png']},
        "table_keywords": {},
        "chart_images": {1: ['output/UOB_2020/3.png', 'output/UOB_2020/4.png'], 2:['output/UOB_2020/5.png']},
        "sentiment_score": {},
        "text_output": {}
    },
]
report_selected = "/"


def getCurrReport():
    print(f"GET: ===> {report_selected}")
    return report_selected


def setCurrReport(report_name):
    global report_selected
    print(f"SET: ===> {report_selected}")
    report_selected = report_name
    print(f"SET: ===> {report_selected}")


def getCurrentYear():
    now = datetime.datetime.now()
    return now.year + 2

# Collect all the files from json input file


def uploaded_files():
    files = []
    for obj in json_input:
        files.append(obj['company']+".pdf")
    # for filename in os.listdir(UPLOAD_DIRECTORY):
    #   path = os.path.join(UPLOAD_DIRECTORY, filename)
    #    if os.path.isfile(path):
    #      files.append(filename)
    return files


files = uploaded_files()  # For all files dir


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    link = "/insights/" + str(filename)
    return html.A(id="selected-report", children=[filename], href=link)


def validateInputs(inputUrl, inputCompany, inputYear):
    message_list = []
    print("==> Validation Starts:")

    if((inputUrl is None)):
        message_list.append("- Please enter a URL. ")
    if((inputCompany is None)):
        message_list.append("- Please enter a company name, e.g. DBS. ")
    if((inputYear is None)):
        message_list.append("- Please select a year. ")
    if(message_list != []):
        return message_list

    pdf_url = str(inputUrl)
    company = str(inputCompany)
    year = str(inputYear)

    # Validation 1: Check if link exist in json
    for obj in json_input:
        print(obj['pdf_url'])
        if(obj['pdf_url'] == pdf_url):
            message_list.append("- Report already exists ")

    # Validation 2: Check if inputCompany exist in json
    for obj in json_input:
        if(obj['company'] == company and obj['year'] == year):
            message_list.append(
                "- Report with company and year already exist ")

    return message_list


def extract_info(pdf_url, pages, path):
    try:
        response = requests.get(pdf_url, timeout=30)
    except:
        print("Requests failed.")
        return "nan"
    print(f"==> Linked Retrieved: {pdf_url}| {path}")
    i = 1
    image_path_obj = {}

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

    print(f"==> Trying process : {company}| {year}| {pdf_url}")
    source = "output/"
    if not os.path.exists(source):  # Check if file exist
        os.mkdir(source)

    path = source + company + '_' + year

    if not os.path.exists(path):
        # file exists
        os.mkdir(path)

    json_obj['images_path'] = extract_info(pdf_url, relevant_pages, path)
    print(f"==> Complete Extraction process")
    json_input.append(json_obj)

    return json_input


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
                dbc.NavItem(dbc.NavLink([icon_insight, "Insights"], href="/insights"+getCurrReport(),
                                        className="nav-Link")),
                dbc.NavItem(dbc.NavLink(
                    [icon_upload, "Upload"], href="/upload", className="nav-Link")),
                dbc.NavItem(dbc.NavLink(
                    [icon_doc, "GitHub"], href="https://github.com/jaokuean/team08-capstone/", className="doc-nav-Link")),
            ],
            vertical=True,
            pills=True,
        ),
    ],
)

# ========================= Main Pages ===========================
# Dashboard Components


def generate_dashboard(report_name):
    dashboard_header = html.Div(
        className="section-header",
        children=[
            html.H6(id="report_name_id", children=[report_name])
        ])

    # TODO: Jermaine
    dashboard_bar_chart = html.Div(
        className="section-dashboard_bar_chart",
        children=[
            html.H6("BAR CHART ")
        ])

    # TODO: Jermaine
    dashboard_wordcloud = html.Div(
        className="section-dashboard_wordcloud",
        children=[
            html.H6("WORDCLOUD ")
        ])

    # TODO: Aifen/Jermainem whomever have more bandwidth
    dashboard_relevant_tables = html.Div(
        className="section-dashboard_relevant_tables",
        children=[
            html.H6("RELEVANT SENTENCES TABLE")
        ])

    # TODO: Aifen
    dashboard_cleaned_tables = html.Div(
        className="section-dashboard_cleaned_tables",
        children=[
            html.H6("CLEANED TABLES EXTRACTED")
        ])

    # TODO: JK
    dashboard_selected_images = html.Div(
        className="section-dashboard_selected_images",
        children=[
            html.H6("SELECTED IMAGES FROM INSIGHTS")
        ])

    # Dashboard layout DIVs
    dashboard_left_content_upper = html.Div(
        className="section-dashboard_left_content_upper",
        children=[
            dashboard_bar_chart,
            dashboard_wordcloud,
        ])
    dashboard_left_content_lower = html.Div(
        className="section-dashboard_left_content_lower",
        children=[
            dashboard_relevant_tables
        ])

    dashboard_left_content = html.Div(
        className="section-dashboard_left_content",
        children=[
            dashboard_left_content_upper,
            dashboard_left_content_lower
        ])

    dashboard_right_content = html.Div(
        className="section-dashboard_right_content",
        children=[
            dashboard_selected_images,
            dashboard_cleaned_tables
        ])

    dashboard = html.Div(
        className="section-main",
        children=[
            dashboard_header,
            dashboard_left_content,
            dashboard_right_content
        ])

    return dashboard


def generate_cards(report_name):
    return


def generate_insight(report_name):
    insights_header = html.Div(
        className="section-header",
        children=[
            html.H6(id="report_name_id", children=[report_name])
        ]
    )

    # Insights Components
    insights_cards = html.Div(
        # For each image extracted from pipeline, output a card
        # Out: Image, Page(title), Extraction type(desc), keywords(badges)
        [
            # generate_cards(report_name)
            dtc.Card(
                className="insights-cards",
                image=app.get_asset_url('output/UOB_2020/1.png'),
                title='Page 35',
                description='Table of Metrics',
                badges=['Carbon Footprint', 'GHG Emissions'],
                style={'display': 'inline-block', "padding": "0.3rem 0.3rem"}
            )
            # for img_path in page for page in json_input['table_images'],
        ]
    )
    insights = html.Div(
        className="section-main",
        children=[
            insights_header,
            insights_cards,
        ]
    )

    return insights


insights_default = html.Div(
    className="section-main",
    children=[
        "Please choose a report in upload library"
    ]
)
# Upload Components
upload_header = html.Div(
    className="section-header",
    children=[
        html.H6("Upload Files")
    ])
upload_content = html.Div(className="upload-main-container", children=[
    html.Div(className="upload-sidebar", children=[
        html.H5("Directory"),
        html.Ul(id="file-list"),
    ]),
    html.Div(className="upload-right-container", children=[
        html.H5(className="upload-sub-header", children=["Upload a PDF file"]),
        dcc.Upload(
            id="upload-pdf",
            className="file-selection-box",
            children=[
                'Drag and Drop or ',
                html.A('Select a File')
            ]),
        html.H4(className="upload-sub-divider",
                children=["OR"], style={'text-align': 'center'}),
        html.H5(className="upload-sub-header",
                children=["Using URL of PDF report"]),
        html.H6(className="upload-sub-header",
                children=[" URL link (.pdf)"]),
        dcc.Input(
            id="inputUrl".format("url"),
            className="upload-right-inputs",
            type='url',
            placeholder="Paste URL link here", size='100',
        ),

        html.Div(id="upload-url-output"),
        html.H6(className="upload-sub-header",
                children=["Company Name"]),
        dcc.Input(
            id="inputCompany".format("text"),
            className="upload-right-inputs",
            type='text',
            placeholder="e.g. DBS", size='45',
        ),
        html.Div(id="upload-companyname-output"),
        html.H6(className="upload-sub-header",
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
        return dcc.Location(pathname="/dashboard/"+getCurrReport(), id="url")
    elif pathname == "/dashboard//":
        return insights_default
    elif "/dashboard" in pathname:
        if(pathname != "/dashboard/"):
            report_name = pathname.split("/")[2]
            setCurrReport(report_name)
            report_curr = getCurrReport()
            # TODO: Link to dashboard main content
            return generate_dashboard(report_curr)
        return insights_default
    elif pathname == "/insights//":
        return insights_default
    elif pathname == "/insights/":
        return dcc.Location(pathname="/insights/"+getCurrReport(), id="url")
    elif "/insights" in pathname:
        # print(f"substring:{pathname[10:]}|{len(pathname)}")
        if(getCurrReport() in pathname or len(pathname) > 10):
            pass
        else:
            pathname = pathname + getCurrReport()
        #print(f"==>current path: {pathname}")
        if(pathname != "/insights/"):
            report_name = pathname.split("/")[2]
            setCurrReport(report_name)
            report_curr = getCurrReport()
            #print(f"==> CURRENT PAGE {report_curr}")
            return generate_insight(report_curr)
        return insights_default
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


# Validation for URL link
# Test Link:
# https://www.uobgroup.com/AR2020/documents/UOB-Sustainability-Report-2020.pdf
# https://wwwap1se.cdn.triggerfish.cloud/uploads/2018/10/ap1_2017_annual_report.pdf

@ app.callback(
    Output("upload-url-output", "children"),
    Input("inputUrl", "value"))
def url_render(inputUrl):
    """
    Valid case:
    1. Contains http
    2. Reachable Link
    3. Unique URL link
    """
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
        # Validation 1: Check if link exist in json
    for obj in json_input:
        print(obj['pdf_url'])
        if(obj['pdf_url'] == inputUrl):
            return u"Report already exists "
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


@ app.callback(
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
    ], prevent_initial_call=True
)
def input_render(n_clicks, inputUrl, inputCompany, inputYear):
    if n_clicks is None:
        raise PreventUpdate

    # Validate that all fields have been entered
    if (n_clicks > 0) and ((inputUrl is None) or (inputCompany is None) or (inputYear is None)):
        return [html.Ol(className="err-list-ol", children=["- Please enter all fields"])]

    print(f"==> Validation Triggered {inputUrl}| {inputCompany}| {inputYear}")
    err_message_list = validateInputs(inputUrl, inputCompany, inputYear)
    print("==> Validation Complete ")

    if(err_message_list == []):
        try:
            output_json = process_pdf_url(inputUrl, inputCompany, inputYear)
            files.append(inputCompany)
            return dcc.Location(pathname="/insights", id="url")
        except Exception as e:
            print(e)
            return [html.Ol(className="err-list-ol", children=["- URL link cannot be converted, try uploading a pdf file instead."])]
    else:
        return [html.Ol(className="err-list-ol", children=[err]) for err in err_message_list]

# Updating file list


@ app.callback(
    Output("file-list", "children"),
    Input('submit-val', 'n_clicks'),
    State("inputCompany", "value"),
)
def update_output(n_clicks, inputCompany):
    # files.append(inputCompany)
    return [html.Ol(className="file-list-li", children=[file_download_link(filename)]) for filename in files]


if __name__ == "__main__":
    app.run_server(debug=True)
