# Latest
# # Data handling prerequisite
import pandas as pd
import plotly.express as px
from collections import OrderedDict
import base64
from io import BytesIO
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from nltk.corpus import stopwords
import os
import datetime
import requests
import json
import pickle5 as pickle
import re

# Dash prerequisites
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State, MATCH, ALL, ALLSMALLER
# import dash_trich_components as dtc
import dash_bootstrap_components as dbc
from dash import dash_table
from wordcloud import WordCloud, STOPWORDS
from dash.exceptions import PreventUpdate
from flask import request

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

report_selected = "/"
report_obj = {}


def file_download_link(company, report_year):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    filename = company + "_" + str(report_year)
    #print("file generated: === > " + filename)
    link = "/insights/" + filename
    return html.A(id="selected-report", children=[filename], href=link)


def getCurrentYear():
    now = datetime.datetime.now()
    return now.year + 2


def uploaded_files():
    files = []
    all_company = []
    for obj in json_input:
        company = obj['company']
        all_company.append(company)

    # Generate unique companies
    unique_company = set(all_company)

    # Declare file structure
    for co in unique_company:
        newfolder = {}
        newfolder['company'] = co
        newfolder['reports'] = []
        files.append(newfolder)

    # Input reports
    for obj in json_input:
        company = obj['company']
        year = obj['year']
        for folder in files:
            if(company == folder['company']):  # match folder
                folder['reports'].append(year)
                folder['reports'].sort(reverse=True)
                break
                # print("ERROR IN READING UNIQUE FOLDERS")
    files.sort(key=lambda x: x["company"])

    return files


def find_report_dict():
    for obj in json_input:
        if obj['company'] == getCurrReport().replace(".pdf", "") and obj['year'] == "2020":  # to change year to a fn
            return obj
    return None


def readData(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    print(f"==> Retrieved DATABASE: {data[0]['company']}{data[0]['year']}")
    return data


def readPickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(
        f"==> Retrieved TABLE PICKLEs: {data[0]['company']}{data[0]['year']}")
    return data


def processMinedText(list):
    newlist = []
    counter = 0
    for item in list:
        try:
            results = re.search('\*(.*)\*', item)
            text_to_bold = results.group(0)
            text_to_bold_last = results.group(0)[-3:]

            if(text_to_bold_last[0] == " "):
                sentence_list = item.split(text_to_bold)
                new_sentence = sentence_list[0] + text_to_bold[:len(
                    text_to_bold)-3] + "** " + sentence_list[1]
                # print(new_sentence)
                newlist.append(new_sentence)
            else:
                newlist.append(item)
        except:
            newlist.append(item)
    return newlist


def processText(data):
    processedData = []
    for obj in data:
        currObj = obj
        currMinedTextList = obj['text_output']['mined_text']
        currObj['text_output']['mined_text'] = processMinedText(
            currMinedTextList)
        processedData.append(currObj)
    return processedData


# Process all database for running dashboard
# 1. loading pickle files
loaded_tbl_all = readPickle('assets/data/tbl_ALL.pickle')

# 2. loading json files
# json_input = readData('assets/data/BlackRock2020.json')
json_input = readData('assets/data/final_database.json')

json_input = processText(json_input)
# 3. Append filelist for directory
files = uploaded_files()  # For all files dir


def getCurrReport():
    print(f"GET: ===> {report_selected}")
    return report_selected


def getCurrReportObj():
    print(f"GET OBJECt: ===> {report_selected}")
    return report_obj


def getUpdatedList(list):
    newlist = []
    if(isinstance(list[0], float) == True):
        for x in list:
            if(x >= 0.8):
                newlist.append("High")
            elif(x >= 0.4):
                newlist.append("Med")
            else:
                newlist.append("Low")
    else:
        return report_obj['text_output']['relevance_prob']
    return newlist


def setCurrReport(report_name):
    global report_selected
    global report_obj

    # Process report name for retrival from database
    report_selected = report_name.replace("%20", " ")
    report_selected = report_selected.replace(".pdf", "")
    report_details = report_selected.split("_")
    company = report_details[0]
    year = report_details[1]

    print(f"SET: ===> {report_selected} | {company} | {year}")
    for obj in json_input:
        # to change year to a fn
        if obj['company'] == company and obj['year'] == year:
            report_obj = obj

            if(report_obj['text_output']['relevance_prob'] == []):
                break
            else:
                report_obj['text_output']['relevance_prob'] = getUpdatedList(
                    report_obj['text_output']['relevance_prob'])

    # Ai Fen: added pickle_input details
    for obj in loaded_tbl_all:
        if obj['company'] == company and obj['year'] == year:
            report_obj['table_dfs'] = obj['tbl_pages']

    # Ai Fen: added kw input details
    for obj in json_input:
        if obj['company'] == company and obj['year'] == year:
            report_obj['table_keywords'] = obj['table_keywords']


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
        if(obj['url'] == pdf_url):
            message_list.append("- Report already exists ")

    # Validation 2: Check if inputCompany exist in json
    for obj in json_input:
        if(obj['company'] == company and obj['year'] == year):
            message_list.append(
                "- Report with company and year already exist ")

    return message_list


def findCompanies(input_value):
    newlist = []
    for company in files:
        if(input_value in company['company'].lower()):
            newlist.append(company)
    return newlist


def generateFilelist(input_value):
    query_files = findCompanies(input_value)
    query_found = len(query_files)
    if(query_found == 0):
        return html.H4("No Search Results")

    file_directory = [
        html.H4(str(query_found) + " Found"),
        html.Div(
            children=[
                html.Ul(
                    html.Div(
                        className="file-list-ul-container",
                        children=[
                            html.H5(children=[filename['company']]),
                            html.Div(
                                className="file-list-li-container",
                                children=[
                                    html.Li(
                                        children=[
                                            file_download_link(
                                                filename['company'], report)
                                        ]
                                    ) for report in filename['reports']
                                ]
                            ),
                        ]
                    )
                )for filename in query_files
            ]
        )
    ]
    return file_directory

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


# ========================= Sidebar ==============================
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
    src=app.get_asset_url('icon-github.png')
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
                        [icon_insight,  html.Span("Insights")], href="/insights"+getCurrReport(), className="nav-Link")
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
            html.H6(id="report_name_id", children=[report_name])
        ])
######################################## DASHBOARD PAGE DONE ########################################################

    dashboard_bar_chart = html.Div(
        className="section-dashboard_bar_chart",
        children=[
            html.H6("Average Sentiment Scores \nacross Categories"),
            dcc.Graph(
                figure={
                    'data': [
                        {
                            'x': ["Carbon Emissions", "Energy", "Waste", "Sustainable Investing", "Others"],
                            'y': report_obj["sentiment_score"],
                            'type': 'bar',
                            'name': 'Category'
                        }
                    ],
                    'layout': {
                        'margin': {'t': 0},
                        "autosize": True,
                    },
                },
            )
        ])

    dashboard_wordcloud1 = html.Div(
        className="section-dashboard_wordcloud1",
        children=[
            html.Img(src=app.get_asset_url(
                report_obj["wordcloud_img_path"][0]))
        ])

    dashboard_wordcloud2 = html.Div(
        className="section-dashboard_wordcloud2",
        children=[
            html.Img(src=app.get_asset_url(
                report_obj["wordcloud_img_path"][1]))
        ])

    dashboard_wordcloud3 = html.Div(
        className="section-dashboard_wordcloud3",
        children=[
            html.Img(src=app.get_asset_url(
                report_obj["wordcloud_img_path"][2]))
        ])

    dashboard_wordcloud4 = html.Div(
        className="section-dashboard_wordcloud4",
        children=[
            html.Img(src=app.get_asset_url(
                report_obj["wordcloud_img_path"][3]))
        ])

    # TODO: Jermaine THIS PART DONE
    dashboard_relevant_tables = html.Div(
        className="section-dashboard_relevant_tables",
        children=[
            html.H6("Decarbonisation-Related Information",
                    style={'display': 'inline-block'}),
            html.Div(
                className="section-dashboard_filter_class",
                children=[

                    html.H4(
                        "Select Carbon Category"
                    ),
                    dcc.Dropdown(
                        id='filter_dropdown',
                        options=[
                            {
                                'label': 'All',
                                'value': 'All'
                            },
                            {
                                'label': 'Carbon Emissions',
                                'value': 'Carbon Emissions'
                            },
                            {
                                'label': 'Energy',
                                'value': 'Energy'},
                            {
                                'label': 'Sustainable Investing',
                                'value': 'Sustainable Investing'
                            },
                            {
                                'label': 'Others',
                                'value': 'Others'
                            }
                        ],
                        value='All'
                    ),
                ]
            ),
            dash_table.DataTable(
                id='table-container',
                columns=[
                    {
                        'name': 'Page No.',
                        'id': 'page',
                        "type": 'text',
                        "presentation": 'markdown'
                    },
                    {
                        'name': 'Sentence',
                        'id': 'mined_text',
                        "type": 'text',
                        "presentation": 'markdown'
                    },
                    {
                        'name': 'Relevance',
                        'id': 'relevance_prob',
                        "type": 'text',
                        "presentation": 'markdown'
                    },
                    {
                        'name': 'Carbon Category',
                        'id': 'carbon_class',
                        "type": 'text',
                        "presentation": 'markdown'
                    }
                ],
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'lineHeight': '25px',
                },
                style_header={
                    'backgroundColor': '#7D55C7',
                    'color': 'white'
                },
                style_cell={
                    'fontSize': 14,
                    'padding': '10px',
                    'font-family': 'Helvetica',
                    'overflow': 'hidden'
                },
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{relevance_prob} contains "High"',
                            'column_id': 'relevance_prob',
                        },
                        'color': 'Green'
                    },
                    {
                        'if': {
                            'filter_query': '{relevance_prob} contains "Med"',
                            'column_id': 'relevance_prob',
                        },
                        'color': 'orange'
                    },
                    {
                        'if': {
                            'filter_query': '{relevance_prob} contains "Low"',
                            'column_id': 'relevance_prob',
                        },
                        'color': 'red'
                    }, {
                        'if': {
                            'filter_query': '{Sentence} eq ""',
                            'column_id': 'Sentence',
                        },
                        'display': 'none'
                    },
                ],
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
            )
        ])
    dashboard_content = html.Div(
        className="grid-dashboard-main",
        children=[
            dashboard_wordcloud1,
            dashboard_wordcloud2,
            dashboard_wordcloud3,
            dashboard_wordcloud4,
            dashboard_bar_chart,
            dashboard_relevant_tables
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
    # print(f"==> GETKEYWORDS : {pagenumber}|{index}")
    keywords_list = []
    for obj in json_input:
        if(obj['company'] == company and obj['year'] == year):
            try:
                keywords_list = obj['table_image_keywords'][pagenumber][index]
            except:
                keywords_list = keywords_list
    # print(f"==> KEYWORDS FOUND : {keywords_list}")
    return keywords_list


def getChartKeywords(company, year, pagenumber, index):
    # print(f"==> GETKEYWORDS : {pagenumber}|{index}")
    keywords_list = []
    for obj in json_input:
        if(obj['company'] == company and obj['year'] == year):
            try:
                keywords_list = obj['chart_images_keywords'][pagenumber][index]
            except:
                keywords_list = keywords_list
    # print(f"==> KEYWORDS FOUND : {keywords_list}")
    return keywords_list


# For getting chart images
def changeFilepath(imgpath):
    # print(imgpath)
    path = imgpath.replace("dashboard", "dashboard_data")
    return path


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
            if(obj['table_images'] == "nan"):
                break
            for page in obj['table_images'].items():
                # print(f"PAGE: {page[1]}")
                for index, img in enumerate(page[1]):
                    count = count + 1
                    # print(f"IMGPATH:  {index} |{img} | {count}")
                    image_path = {
                        "imagepath": img,
                        "pageNum": page[0],
                        "ext_type": "Table Image",
                        "keywords": getTblKeywords(report_company_name, report_year, page[0], index),
                        "image_count": count,
                    }
                    # print(f"{count}image found on page {page[0]}: {image_path}")
                    card_list.append(image_path)
            break

    # Get Chart images
    for obj in json_input:
       # print(f"OBJ: {obj['table_images'].items()}")
        if(obj['company'] == report_company_name and obj['year'] == report_year):
            # print(f"entered: {obj['chart_images']}")
            if(obj['chart_images'] == "nan"):
                break
            for page in obj['chart_images'].items():
                # print(f"PAGE: {page[1]}")
                for index, img in enumerate(page[1]):
                    count = count + 1
                    # print(f"IMGPATH:  {index} |{img} | {count}")
                    image_path = {
                        "imagepath": changeFilepath(img),
                        "pageNum": page[0],
                        "ext_type": "Chart Image",
                        "keywords": getChartKeywords(report_company_name, report_year, page[0], index),
                        "image_count": count,
                    }
                    # print(f"{count}image found on page {page[0]}: {image_path}")
                    card_list.append(image_path)
            break

    # print(card_list)
    return card_list


def generate_df_cards():
    df_card_lst = []
    count = 1  # counter for df
    dropdown_vals = ['All', ]

    # gather keywords
    print(f"entered: {report_obj['table_keywords']}")
    print(f"entered2: {report_obj}")
    if(report_obj['table_keywords'] == "nan"):
        pass
    else:
        for page, kws in report_obj['table_keywords'].items():
            if len(kws) == 0:
                continue
            else:
                for kw in kws:
                    # list of keyword per table on page
                    kw = list(map(lambda x: x.capitalize(), kw))
                    df_kw = {
                        "table_count": count,
                        "keywords": kw,
                        "pageNum": page
                    }
                    df_card_lst.append(df_kw)
                    dropdown_vals.extend(kw)
                    count += 1

        # gather dfs
        idx = 0
        for page, tbls in report_obj['table_dfs'].items():
            if len(tbls) == 0:
                continue
            else:
                for tbl in tbls:
                    df_col_param = []
                    for col in tbl.columns:
                        df_col_param.append({"name": str(col), "id": str(col)})
                    df_card_lst[idx]['df'] = tbl.to_dict('records')
                    df_card_lst[idx]['df_cols'] = df_col_param
                    idx += 1

    # print(df_card_lst)
    return df_card_lst, sorted(list(set(dropdown_vals)))


def generate_insight(report_name, card_list):
    insights_header = html.Div(
        className="section-header",
        children=[
            html.H6(id="report_name_id", children=[report_name])
        ]
    )
    df_cards, dropdown_vals = generate_df_cards()
    df_cards_full = [
        html.Div(
            className="insights-grid",
            children=[
                dbc.Card(
                      className="insights-cards",
                      children=[
                          dbc.CardBody(
                                [
                                    dash_table.DataTable(
                                        id='cleaned-tables-container',
                                        columns=df['df_cols'],
                                        data=df['df'],
                                        style_data={
                                            'whiteSpace': 'normal',
                                            'height': 'auto',
                                            'lineHeight': '25px',
                                        },
                                        style_header={
                                            'backgroundColor': '#7D55C7',
                                            'color': 'white'
                                        },
                                        style_cell={
                                            'fontSize': 14,
                                            'padding': '10px',
                                            'font-family': 'Helvetica',
                                            'overflow': 'hidden'
                                        },
                                    ),
                                    html.H4(
                                        "Retrieved from Page " + df['pageNum'],
                                        className="insights-card-title"
                                    ),
                                    html.P(
                                        className="insights-card-footer",
                                        children=[
                                            html.Span(
                                                children=[
                                                    word,
                                                ]
                                            ) for word in df['keywords']
                                        ]
                                    ),
                                ]
                          ),
                      ]
                ) for df in df_cards
            ]
        )
    ]

    insights_charts = html.Div(
        # For each image extracted from pipeline, output a card
        # Out: Image, Page(title), Extraction type(desc), keywords(badges)
        className="insights_left_content",
        children=[
            html.H5("Decarbonisation-Related Chart and Figures"),
            html.H6(str(len(card_list))+" Images Found"),
            dcc.Dropdown(
                id='filter-dropdown-pd',
                options=[
                    {
                        'label': kw, 'value': kw
                    } for kw in dropdown_vals
                ], value='All'
            ),
            html.Div(
                className="insights-grid",
                children=[
                    dbc.Card(
                        className="insights-cards",
                        children=[
                            dbc.CardBody(
                                [
                                    html.Img(
                                        className="insights-card-img",
                                        src=app.get_asset_url(img['imagepath'])
                                    ),
                                    html.H4(
                                        "Retrieved from Page " + \
                                        img['pageNum'],
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
                ]
            )
        ],
    )

    insights_tables = html.Div(
        className="insights_right_content",
        id="insights_cleaned_tables",
        children=[
            html.H5("Decarbonisation-Related Tables"),
            html.H6(str(len(df_cards))+" Tables Found"),
            dcc.Dropdown(
                id='filter-dropdown-pd',
                options=[
                    {
                        'label': kw, 'value': kw
                    } for kw in dropdown_vals
                ], value='All'
            )
        ] + df_cards_full
    )

    insights_content = html.Div(
        className="insights-content",
        children=[
            insights_charts,
            insights_tables
        ]
    )
    insights = html.Div(
        className="section-main",
        children=[
            insights_header,
            insights_content,

        ]
    )

    return insights


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
                dcc.Input(
                    id='searchquery',
                    placeholder="Search Report by Company",
                    value='',
                    type='text'
                ),
                html.Div(id="file-list"),
            ]
        ),
        html.Div(
            className="upload-right-container",
            children=[
                html.H5(
                    className="upload-sub-header-top",
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
                    className="upload-sub-header-top",
                    children=["Paste URL of PDF report"]
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

    ]
)
upload_fn = html.Div(
    className="section-main",
    children=[
        upload_header,
        upload_content
    ]
)

# Default Page when user have not selected a report from library
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
    # DASHBOARD
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
    # INSIGHTS
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
    # UPLOAD/LIBRARY
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
        if(obj['url'] == inputUrl):
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
            new_url, new_company, new_year = inputUrl, inputCompany, inputYear
            print("====> NEW REPORT TO ADD: ")
            print(new_url, new_company, new_year)
            filetoappend = f"{new_company}_{new_year}.pdf"
            return dcc.Location(pathname="/insights/"+filetoappend, id="url")
        except Exception as e:
            print(e)
            return [html.Ol(className="err-list-ol", children=["- URL link cannot be processed."])]
    else:
        return [html.Ol(className="err-list-ol", children=[err]) for err in err_message_list]

# Updating file list


@ app.callback(
    Output("file-list", "children"),
    Input('searchquery', 'value'),
)
def update_output(input_value):
    # print(f'SEARCHING ====>>>> :|{input_value}|')
    updateList = generateFilelist(input_value)

    return updateList

    # Callback for filtering of lst of pandas tables


@ app.callback(
    Output('insights_cleaned_tables', 'children'),
    [Input('filter-dropdown-pd', 'value')])
def display_filtered_pd_table(state):
    df_cards, dropdown_vals = generate_df_cards()
    df_cards_full = [
        dbc.Card(
            className="insights-cards",
            children=[
                dbc.CardBody(
                    [
                        dash_table.DataTable(
                            id='cleaned-tables-container',
                            columns=df['df_cols'],
                            data=df['df'],
                            style_data={
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'lineHeight': '25px',
                            },
                            style_header={
                                'backgroundColor': '#7D55C7',
                                'color': 'white'
                            },
                            style_cell={
                                'fontSize': 14,
                                'padding': '10px',
                                'font-family': 'Helvetica',
                                'overflow': 'hidden'
                            },
                        ),
                        html.H4(
                            "Retrieved from Page " + df['pageNum'],
                            className="insights-card-title"
                        ),
                        html.P(
                            className="insights-card-footer",
                            children=[
                                html.Span(
                                    children=[
                                        word,
                                    ]
                                ) for word in df['keywords']
                            ]
                        ),
                    ]
                ),
            ]
        ) for df in df_cards
    ]

    unfiltered_df = [
        html.H5("Decarbonisation-Related Tables"),
        html.H6(str(len(df_cards))+" Tables Found"),
        dcc.Dropdown(
            id='filter-dropdown-pd',
            options=[
                {
                    'label': kw,
                    'value': kw
                } for kw in dropdown_vals
            ], value=state
        )
    ] + df_cards_full

    if state == 'All':
        return unfiltered_df
    else:
        idxs = []
        for idx, tbl in enumerate(df_cards):
            if state in tbl['keywords']:
                idxs.append(idx+2)

    filtered_df = unfiltered_df[0:2]
    for i in idxs:

        filtered_df.append(unfiltered_df[i])

    return filtered_df

# Callback for filtering of carbon class
# 36 and 3rd last


@ app.callback(
    Output('table-container', 'data'),
    [Input('filter_dropdown', 'value')])
def display_table(state):

    df = pd.DataFrame.from_dict({
        x: report_obj['text_output'][x] for x in report_obj['text_output'] if x != "sentence"
    })

    if state == 'All':
        return df.to_dict('records')
    else:
        dff = df[df.carbon_class == state]
        return dff.to_dict('records')


if __name__ == "__main__":
    app.run_server(debug=True)
