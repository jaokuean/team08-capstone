# Dash prerequisites
import os
import datetime
import requests
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State, MATCH, ALL, ALLSMALLER
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
        "wordcloud_img_path": "wordcloud_images/AIIB_2020.png",
        "table_images": {1: ['output/ASAHI_2020/1.png', 'output/ASAHI_2020/2.png'], 2:['output/ASAHI_2020/2.png']},
        "table_keywords": {},
        "chart_images": {1: ['output/ASAHI_2020/3.png', 'output/ASAHI_2020/4.png'], 2:['output/ASAHI_2020/5.png']},
        "sentiment_score": [0.85, 0.5, 0.3, 0.9, 0.76],
        "text_output": {
            "page": [4, 4, 5, 6],
            "sentence": ["sentence1", "sentence2", "sentence3", "sentence4"],
            "relevance_prob": [97, 93, 73, 64],
            "carbon_class": ["Emissions", "Renewables", "Sustainable Investing", "Renewables"],
            "mined_text": ["text1", "text2", "text3", "text4"]
        }
    },
    {
        "company": "UOB",
        "year": "2020",
        "pdf_url": "https://www.uobgroup.com/AR2020/documents/UOB-Sustainability-Report-2020.pdf",
        "wordcloud_img_path": "wordcloud_images/OCBC Bank_2020.png",
        "table_images": {1: ['output/UOB_2020/1.png', 'output/UOB_2020/2.png'], 2:['output/UOB_2020/2.png']},
        "table_keywords": {},
        "chart_images": {1: ['output/UOB_2020/3.png', 'output/UOB_2020/4.png'], 2:['output/UOB_2020/5.png']},
        "sentiment_score": [0.65, 1, 0.8, 0.9, 0.92],
        "text_output": {
            "page": [4, 4, 5, 6],
            "sentence": ["At the end of 2019, we had reduced our GHG emissions by 71% compared to baseline year 2004.",
                         "These efforts reduced energy consumption by more than 2,100 metric tons (mt) of carbon dioxide equivalents (CO2e) during the one-year challenge.",
                         "For public and private assets, excluding cash and non-equity derivatives as they were not reported in 2019, our year-over-year portfolio weighted average carbon intensity was reduced by approximately 23%.",
                         "sentence4"],
            "relevance_prob": [97, 93, 73, 64],
            "carbon_class": ["Emissions", "Renewables", "Sustainable Investing", "Renewables"],
            "mined_text": ["text1", "text2", "text3", "text4"]
        }
    },
]
report_selected = "/"
report_obj = {}


def getCurrReport():
    return report_selected


def getCurrReportObj():
    return report_obj


def setCurrReport(report_name):
    global report_selected
    global report_obj
    print(f"SET: ===> {report_selected}")
    report_selected = report_name
    print(f"SET: ===> {report_selected}")
    for obj in json_input:
        # to change year to a fn
        if obj['company'] == report_selected.replace(".pdf", "") and obj['year'] == "2020":
            report_obj = obj


def getCurrentYear():
    now = datetime.datetime.now()
    return now.year + 2

# Collect all the files from json input file


def uploaded_files():
    files = []
    for obj in json_input:
        files.append(f"{obj['company']}_{obj['year']}.pdf")
    # for filename in os.listdir(UPLOAD_DIRECTORY):
    #   path = os.path.join(UPLOAD_DIRECTORY, filename)
    #    if os.path.isfile(path):
    #      files.append(filename)
    return files


def find_report_dict():
    for obj in json_input:
        if obj['company'] == getCurrReport().replace(".pdf", "") and obj['year'] == "2020":  # to change year to a fn
            return obj
    return None


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
        if(obj['pdf_url'] == pdf_url):
            message_list.append("- Report already exists ")

    # Validation 2: Check if inputCompany exist in json
    for obj in json_input:
        if(obj['company'] == company and obj['year'] == year):
            message_list.append(
                "- Report with company and year already exist ")

    return message_list

# TODO: PIPELINE CODE HERE


def extract_info(pdf_url, pages, path):
    try:
        response = requests.get(pdf_url, timeout=30)
    except:
        print("Requests failed.")
        return "nan"
    i = 1
    image_path_obj = {}

    images = convert_from_bytes(response.content)
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
icon_db = html.Img(
    className="icon-nav-link",
    src=app.get_asset_url('icon-dashboard.png')
)
icon_insight = html.Img(
    className="icon-nav-link",
    src=app.get_asset_url('icon-insights.png')
)
icon_upload = html.Img(
    className="icon-nav-link",
    src=app.get_asset_url('icon-doc.png')
)
icon_doc = html.Img(
    className="icon-nav-link",
    src=app.get_asset_url('git-logo.png')
)
left_content = html.Div(
    className="sidebar",
    children=[
        html.Img(className="logo",
                 src='https://creativereview.imgix.net/content/uploads/2016/10/NW_logo_still_800px.jpg'),
        dbc.Nav(
            [
                dbc.NavItem(
                    dbc.NavLink(
                        [icon_db, html.Span("Dashboard")], href="/dashboard", className="nav-Link")
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        [icon_insight,  html.Span("Insights")], href="/insights" + getCurrReport(), className="nav-Link")
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        [icon_upload,  html.Span("Library")], href="/upload", className="nav-Link")
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        [icon_doc,  html.Span("GitHub")], href="https://github.com/jaokuean/team08-capstone/", className="doc-nav-Link")
                ),
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
            html.H6(children=[report_name])
        ])

    # TODO: Jermaine
    dashboard_bar_chart = html.Div(
        className="grid-item item1",
        children=[
            html.H6("Average Sentiment Scores Across Categories"),
            dcc.Graph(
                figure={
                    'data': [
                        {'x': ["Emissions", "Renewables", "Waste", "Sustainable Investing", "Others"],
                         'y': report_obj["sentiment_score"],
                         'type': 'bar', 'name': 'Category'}
                    ],
                    'layout': {
                        'margin': {'t': 10},
                        "autosize": True,
                    }
                }, style={'height': '330px'}
            )
        ])

    # TODO: Jermaine
    dashboard_wordcloud = html.Div(
        className="grid-item item2",
        children=[
            html.H6("Word Cloud of Top 10 Keywords Per Topic"),
            html.Img(src=app.get_asset_url(report_obj["wordcloud_img_path"]))
        ])

    # TODO: Aifen/Jermainem whomever have more bandwidth
    dashboard_relevant_tables = html.Div(
        className="grid-item item3",
        children=[
            html.H6("Decarbonisation-Related Information",
                    style={'display': 'inline-block'}),
            html.Div(
                className="section-dashboard_filter_class",
                children=[
                    "Select Carbon Category",
                    dcc.Dropdown(
                        id='filter_dropdown',
                        options=[{'label': 'All', 'value': 'All'},
                                 {'label': 'Emissions', 'value': 'Emissions'},
                                 {'label': 'Renewables', 'value': 'Renewables'},
                                 {'label': 'Sustainable Investing',
                                     'value': 'Sustainable Investing'},
                                 {'label': 'Others', 'value': 'Others'}
                                 ],
                        value='All'
                    ),
                ]
            ),
            dash_table.DataTable(
                id='table-container',
                columns=[{'name': 'Page No.', 'id': 'page'},
                         {'name': 'Sentence', 'id': 'sentence'},
                         {'name': 'Relevance (%)', 'id': 'relevance_prob'},
                         {'name': 'Carbon Category', 'id': 'carbon_class'}
                         ],
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_header={
                    'backgroundColor': '#D71C2B',
                    'color': 'white'
                },
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
            )
        ])

    # TODO: Aifen
    dashboard_cleaned_tables = html.Div(
        className="grid-item item4",
        children=[
            html.H6("CLEANED TABLES EXTRACTED")
        ])

    # TODO: JK
    dashboard_selected_images = html.Div(
        className="grid-item item5",
        children=[
            html.H6("SELECTED IMAGE FROM INSIGHTS"),
            html.Div(
                className="dashboard-cards-container",
                children=[
                    dbc.Card(
                        className="dashboard-cards",
                        children=[
                            dbc.CardBody(
                                children=[
                                    html.H4(
                                        "Page " + img['pageNum'],
                                        className="insights-card-title"
                                    ),
                                    html.Img(
                                        className="dashboard-card-img",
                                        src=app.get_asset_url(img['imagepath'])
                                    ),
                                ]
                            )for img in getImgInsightsDashboard()
                        ]
                    )
                ]
            )
        ]
    )

    dashboard_content = html.Div(
        className="grid-dashboard-main",
        children=[
            dashboard_bar_chart,
            dashboard_wordcloud,
            dashboard_cleaned_tables,
            dashboard_selected_images,
            dashboard_relevant_tables,
        ]
    )
    dashboard = html.Div(
        className="section-main",
        children=[
            dashboard_header,
            dashboard_content
        ]
    )
    return dashboard


# Insights Components
def getTblKeywords(company, year, pagenumber, index):
    #print(f"==> GETKEYWORDS : {pagenumber}|{index}")
    keywords_list = []
    for obj in json_input:
        if(obj['company'] == company and obj['year'] == year):
            try:
                keywords_list = obj['table_keywords'][pagenumber][index]
            except:
                keywords_list = keywords_list
    #print(f"==> KEYWORDS FOUND : {keywords_list}")
    return keywords_list


def getChartKeywords(company, year, pagenumber, index):
    #print(f"==> GETKEYWORDS : {pagenumber}|{index}")
    keywords_list = []
    for obj in json_input:
        if(obj['company'] == company and obj['year'] == year):
            try:
                keywords_list = obj['chart_keywords'][pagenumber][index]
            except:
                keywords_list = keywords_list
    #print(f"==> KEYWORDS FOUND : {keywords_list}")
    return keywords_list


def getSelectedCards():
    n_clicks = []
    report_name = getCurrReport()
    report_name = report_name.replace(".pdf", "")
    report_company_name = report_name.split("_")[0]
    report_year = report_name.split("_")[1]
    # print(f"==> REPORT IN: {report_name}| {report_company_name}| {report_year}")
    for index, clickcount in enumerate(n_clicks):
        if(clickcount % 2 == 0):
            n_clicks[index] = 0
        else:
            n_clicks[index] = 1
    print(f"==> CLICKCOUNT ALL : {n_clicks}")

    for obj in json_input:
       # print(f"OBJ: {obj['table_images'].items()}")
        if(obj['company'] == report_company_name and obj['year'] == report_year):
            n_clicks = obj['display_dashboard']
    return n_clicks


def updateSelectedCards(n_clicks):
    report_name = getCurrReport()
    report_name = report_name.replace(".pdf", "")
    report_company_name = report_name.split("_")[0]
    report_year = report_name.split("_")[1]
    # print(f"==> REPORT IN: {report_name}| {report_company_name}| {report_year}")
    for index, clickcount in enumerate(n_clicks):
        if(clickcount % 2 == 0):
            n_clicks[index] = 0
        else:
            n_clicks[index] = 1
    print(f"==> CLICKCOUNT ALL : {n_clicks}")

    for obj in json_input:
       # print(f"OBJ: {obj['table_images'].items()}")
        if(obj['company'] == report_company_name and obj['year'] == report_year):
            obj['display_dashboard'] = n_clicks

    return sum(n_clicks)


def generate_cards(report_name):
    card_list = []
    count = 0  # counter for images
    report_name = report_name.replace(".pdf", "")
    report_company_name = report_name.split("_")[0]
    report_year = report_name.split("_")[1]

    # Get table images
    for obj in json_input:
       # print(f"OBJ: {obj['table_images'].items()}")
        if(obj['company'] == report_company_name and obj['year'] == report_year):
            for page in obj['table_images'].items():
                # print(f"PAGE: {page[1]}")
                for index, img in enumerate(page[1]):
                    count = count + 1
                    #print(f"IMGPATH:  {index} |{img} | {count}")
                    image_path = {
                        "imagepath": img,
                        "pageNum": page[0],
                        "ext_type": "Table Image",
                        "keywords": getTblKeywords(report_company_name, report_year, page[0], index),
                        "image_count": count,
                        "n_click": obj['display_dashboard'][count-1],
                    }
                    # print(f"{count}image found on page {page[0]}: {image_path}")
                    card_list.append(image_path)
            break

    # Get Chart images
    for obj in json_input:
       # print(f"OBJ: {obj['table_images'].items()}")
        if(obj['company'] == report_company_name and obj['year'] == report_year):
            for page in obj['chart_images'].items():
                # print(f"PAGE: {page[1]}")
                for index, img in enumerate(page[1]):
                    count = count + 1
                    #print(f"IMGPATH:  {index} |{img} | {count}")
                    image_path = {
                        "imagepath": img,
                        "pageNum": page[0],
                        "ext_type": "Chart Image",
                        "keywords": getChartKeywords(report_company_name, report_year, page[0], index),
                        "image_count": count,
                        "n_click": obj['display_dashboard'][count-1],
                    }
                    #print(f"{count}image found on page {page[0]}: {image_path}")
                    card_list.append(image_path)
            break

    # print(card_list)
    return card_list


def generate_insight(report_name, card_list):
    insights_header = html.Div(
        className="section-header",
        children=[
            html.H6(id="report_name_id", children=[report_name])
        ]
    )

    insights_cards = html.Div(
        # For each image extracted from pipeline, output a card
        # Out: Image, Page(title), Extraction type(desc), keywords(badges)
        className="insights-grid",
        children=[

            # generate_cards(report_name)
            dbc.Card(
                className="insights-cards",
                children=[
                    dbc.CardBody(
                        [
                            html.Button(
                                className="insights-cards-button",
                                children=[''],
                                id={
                                    'type': 'insights-select-card',
                                    'index': img['image_count']
                                },
                                n_clicks=img['n_click'],
                                disabled=False,
                            ),
                            html.Img(
                                className="insights-card-img",
                                src=app.get_asset_url(img['imagepath'])
                            ),
                            html.H4(
                                "Page " + img['pageNum'],
                                className="insights-card-title"
                            ),
                            html.P(
                                img['ext_type'],
                                className="insights-card-text"
                            ),
                            html.P(
                                className="insights-card-footer",
                                children=[
                                    html.Span(
                                        children=[
                                            word,
                                        ]
                                    ) for word in img['keywords']
                                ]
                            ),
                        ]
                    ),
                ]
            )for img in card_list
        ],
    )

    insights = html.Div(
        className="section-main",
        children=[
            insights_header,
            html.H6(id="selected_cards"),
            insights_cards,
        ]
    )

    return insights


@app.callback(
    Output('selected_cards', 'children'),
    Input({'type': 'insights-select-card', 'index': ALL}, 'n_clicks')
)
def display_output(n_clicks):
    if(sum(n_clicks) == 0):
        n_clicks = getSelectedCards()
    print(n_clicks)
    total_selected = updateSelectedCards(n_clicks)
    return f"You have selected {total_selected} images."


default_page = html.Div(
    className="section-main",
    children=[
        html.Div(
            className="default-main",
            children=[
                html.H5("Please choose a report from library"),
                dcc.Link(html.Button('Go to Library'), href="/upload")
            ])
    ]
)
# Upload Components
upload_header = html.Div(
    className="section-header",
    children=[
        html.H6("Upload Files")
    ]
)
upload_content = html.Div(
    className="upload-main-container",
    children=[
        html.Div(
            className="upload-sidebar",
            children=[
                html.H5("Directory"),
                html.Ul(id="file-list"),
            ]
        ),
        html.Div(
            className="upload-right-container",
            children=[
                html.H5(
                    className="upload-sub-header",
                    children=["Upload a PDF file"]
                ),
                dcc.Upload(
                    id="upload-pdf",
                    className="file-selection-box",
                    children=[
                        'Drag and Drop or ',
                        html.A('Select a File')
                    ]
                ),
                html.H4(
                    className="upload-sub-divider",
                    children=["OR"],
                    style={'text-align': 'center'}
                ),
                html.H5(
                    className="upload-sub-header",
                    children=["Using URL of PDF report"]
                ),
                html.H6(
                    className="upload-sub-header",
                    children=[" URL link (.pdf)"]
                ),
                dcc.Input(
                    id="inputUrl".format("url"),
                    className="upload-right-inputs",
                    type='url',
                    placeholder="Paste URL link here",
                    size='100'
                ),
                html.Div(
                    className="err-message", id="upload-url-output"
                ),
                html.H6(
                    className="upload-sub-header",
                    children=["Company Name"]
                ),
                dcc.Input(
                    id="inputCompany".format("text"),
                    className="upload-right-inputs",
                    type='text',
                    placeholder="e.g. DBS",
                    size='45',
                ),
                html.Div(
                    className="err-message",
                    id="upload-companyname-output"
                ),
                html.H6(
                    className="upload-sub-header",
                    children=["Year of Report"]
                ),
                dcc.Dropdown(
                    id="inputYear",
                    placeholder="Select Year",
                    options=[
                        {'label': x, 'value': x}
                        for x in reversed(range(1999, getCurrentYear()))
                    ],
                    style={
                        'border': "none",
                        'margin-top': '10px',
                        'border-bottom': '1px solid #adadad'
                    }
                ),
                html.Div(
                    className="err-message",
                    id="upload-year-output"),
                html.Button(
                    className="upload-right-button",
                    children=['Submit'],
                    id='submit-val',
                    n_clicks=0
                ),
                html.Div(id="upload-output"),
            ]),

    ])
upload_fn = html.Div(
    className="section-main",
    children=[
        upload_header,
        upload_content
    ])


# ========================= App Layout ===========================
right_content = html.Div(
    id="page-content"
)

# Landing page when user first enter
index = html.Div(
    dcc.Link('upload', refresh=True, href='/upload')
)

app.layout = html.Div(
    className='content', children=[
        dcc.Location(id="url", refresh=True), left_content, right_content
    ]
)

# ========================= Back end callbacks ===========================
# Callback for NavBar


@ app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return index
    elif pathname == "/dashboard":
        return dcc.Location(pathname="/dashboard/"+getCurrReport(), id="url")
    elif pathname == "/dashboard//":
        return default_page
    elif "/dashboard" in pathname:
        if(pathname != "/dashboard/"):
            report_name = pathname.split("/")[2]
            setCurrReport(report_name)
            report_curr = getCurrReport()
            return generate_dashboard(report_curr)
        return default_page
    elif pathname == "/insights//":
        return default_page
    elif pathname == "/insights/":
        return dcc.Location(pathname="/insights/"+getCurrReport(), id="url")
    elif "/insights" in pathname:
        # print(f"substring:{pathname[10:]}|{len(pathname)}")
        if(getCurrReport() in pathname or len(pathname) > 10):
            pass
        else:
            pathname = pathname + getCurrReport()
        # print(f"==>current path: {pathname}")
        if(pathname != "/insights/"):
            report_name = pathname.split("/")[2]
            setCurrReport(report_name)
            report_curr = getCurrReport()
            print(f"==> CURRENT PAGE {report_curr}")
            card_list = generate_cards(report_curr)
            return generate_insight(report_curr, card_list)
        return default_page
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

# Callback for filtering of carbon class


@app.callback(
    Output('table-container', 'data'),
    [Input('filter_dropdown', 'value')])
def display_table(state):
    df = pd.DataFrame.from_dict({
        x: report_obj['text_output'][x] for x in report_obj['text_output'] if x != "mined_text"
    })
    if state == 'All':
        return df.to_dict('records')
    else:
        dff = df[df.carbon_class == state]
        return dff.to_dict('records')

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
            filetoappend = f"{inputCompany}_{inputYear}.pdf"
            print(filetoappend)
            files.append(filetoappend)
            return dcc.Location(pathname="/insights/"+filetoappend, id="url")
        except Exception as e:
            print(e)
            return [html.Ol(className="err-list-ol", children=["- URL link does not contain a pdf file."])]
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
