# Dash components
from flask import request
import dash
from dash.exceptions import PreventUpdate
from wordcloud import WordCloud, STOPWORDS
from dash import dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL, ALLSMALLER
from dash import html
from dash import dcc

# Data handling prerequisite
import re
import pickle5 as pickle
import json
import requests
import datetime
import os
from nltk.corpus import stopwords
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from main import *
import pandas as pd
from collections import OrderedDict
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'NatWest Dashboard'

# ========================= ALL METHODS FOR DASHBOARD ===========================
report_selected = "/"
report_obj = {}


def generate_file_links(company, report_year):
    """
    Generate a html <a> link for each report in the database.

    Parameters
    ----------
    company: str
        Company Report Name.
    report_year: str
        Year of report.

    Return
    ------
    Html component <a>  for each report.
    """
    filename = company + "_" + str(report_year)
    link = "/insights/" + filename
    return html.A(id="selected-report", children=[filename], href=link)


def getDropdownYears():
    """
    Get valid years for input in library page.

    Parameters
    ----------

    Return
    ------
    list of int
        List of years input for dropdown plus two years ahead.
    """
    now = datetime.datetime.now()
    return now.year + 2


def generate_reports():
    """
    Generate a list of reports with object data retrieved from the final_database.json file stored in json_input.

    Parameters
    ----------

    Return
    ------
    files : list of str
        List of sorted reports grouped by company name and years.
    """
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
    files.sort(key=lambda x: x["company"])

    return files


def readData(file_path):
    """
    Read json file with all data extraction pipeline.

    Parameters
    ----------
    file_path: str
        Image path to json file.

    Return
    ------
    data : dict of {str : str or dict}
        Dictionary containing company's report details, preprocessed text and table/chart pipeline output.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Check if json is loaded properly
    print(f"==> Retrieved DATABASE JSON FILE: {len(data)} records found!")
    return data


def readPickle(file_path):
    """
    Read pickle file for cleaned table dataframes from table extraction pipeline.

    Parameters
    ----------
    file_path: str
        Image path to pickle file.

    Return
    ------
    data : dict of {str : str or dict}
        Dictionary containing table pipeline output.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(
        f"==> Retrieved TABLE PICKLE DATA: {len(data)} records found1")
    return data


def processMinedText(list):
    """
    Method to process every relevant sentences extracted mined text and ensure that bold markdown works correctly by adjusting "**" in between a set of relevant sentences.

    Parameters
    ----------
    list: list of str
        List of mined text from database json file.

    Return
    ------
    newlist : list of str
        List of mined text with correct markdowns.
    """
    newlist = []
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
    """
    Method to process each report object for to ensure correct formatting.

    Parameters
    ----------
    data: dict of {str : str or dict}
        List of releveant sentences from database json file

    Return
    ------
    processedData : dict of {str : str or dict}
        Dictionary containing formatted company's report details, preprocessed text and table pipeline output.
    """
    processedData = []
    for obj in data:
        currObj = obj
        currMinedTextList = obj['text_output']['mined_text']
        currObj['text_output']['mined_text'] = processMinedText(
            currMinedTextList)
        processedData.append(currObj)
    return processedData


def getCurrReport():
    """
    Get the current report that user has selected from library page.

    Parameters
    ----------

    Return
    ------
    report_selected : str
        File name of report (<COMPANY>_<YEAR>.pdf)
    """
    print(f"CURRENT REPORT: ===> {report_selected}")
    return report_selected


def generate_relevance_cat(list):
    """
    Method to process each relevance_prob of each extracted sentences for each report and convert them to high, med and low for viewing in the dashboard.

    Parameters
    ----------
    list: list of str
        List of relevance probability of each relevant sentences.

    Return
    ------
    newlist : list of str
        List of relevance probability of a relevant sentence extracted.
    """
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
    """
    Method to collect selected report data from user in Library page.

    Parameters
    ----------
    report_name: str
        Report name selected by user.

    Return
    ------
    """
    global report_selected
    global report_obj

    # Process report name for retrival from database json file
    report_selected = report_name.replace("%20", " ")
    report_selected = report_selected.replace(".pdf", "")
    report_details = report_selected.split("_")
    company = report_details[0]
    year = report_details[1]

    print(
        f"USER SELECTED NEW REPORT: ===> {report_selected} | {company} | {year}")
    for obj in json_input:
        if obj['company'] == company and obj['year'] == year:
            report_obj = obj

            if(report_obj['text_output']['relevance_prob'] == []):
                break
            else:
                report_obj['text_output']['relevance_prob'] = generate_relevance_cat(
                    report_obj['text_output']['relevance_prob'])

    # Added pickle file details for clean tables dataframe to each report data object
    for obj in loaded_tbl_all:
        if obj['company'] == company and obj['year'] == year:
            report_obj['table_dfs'] = obj['tbl_pages']

    # Added pickle file details for clean tables dataframe corresponding keywords to each report data object
    for obj in json_input:
        if obj['company'] == company and obj['year'] == year:
            report_obj['table_keywords'] = obj['table_keywords']


def validateInputs(inputUrl, inputCompany, inputYear):
    """
    Method to validate users input. Each error message is returned when the conditions are met.
    1. User did not enter any of the input fields.
    2. User enter a pdf url link that already exist in our database.
    3. User enter a company report name with the same year.

    Parameters
    ----------
    inputUrl: str
        Report url input by user.
    inputCompany: str
        Report company input by user.
    inputYear: int
        Report year input by user.

    Return
    ------
    message_list : list of str
        List of error messages based on user input.
    """
    message_list = []
    print("==> Validation Starts:")

    # Validation 1: Check if user entered respective fields
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

    # Validation 3: Check if link exist in json
    for obj in json_input:
        if(obj['url'] == pdf_url):
            message_list.append("- Report already exists ")

    # Validation 4: Check if inputCompany exist in json
    for obj in json_input:
        if(obj['company'] == company and obj['year'] == year):
            message_list.append(
                "- Report with company and year already exist ")

    return message_list


def findCompanies(input_value):
    """
    Method to search for a report based on search query.

    Parameters
    ----------
    input_value: str
        Search query input by user in search bar.

    Return
    ------
    newlist : list of str
        List of reports found by search query input.
    """
    newlist = []
    for company in files:
        if(input_value.lower() in company['company'].lower()):
            newlist.append(company)
    return newlist


def generateSearchFileList(input_value):
    """
    Method to generate components for directory after user input a search query.

    Parameters
    ----------
    input_value: str
        Search query input by user in search bar.

    Return
    ------
    file_directory : list of dash components
        List of dash components based on user search input value.
    """
    query_files = findCompanies(input_value)
    query_found = len(query_files)
    if(query_found == 0):
        return html.H4("No Search Results")

    total_report = 0
    for company in query_files:
        total_report = total_report + len(company['reports'])

    file_directory = [
        html.H4(str(query_found) + " Companies Found"),
        html.H4(str(total_report) + " Reports Found"),
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
                                            generate_file_links(
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


def run_extraction_pipeline(pdf_url, report_company, report_year):
    """
    Method to generate components for directory after user input a search query.

    Parameters
    ----------
    pdf_url: str
        Valid pdf url input by user in upload section.
    report_company: str
        Valid report name input by user in upload section.
    report_year: int
        Valid report year input by user in upload section.

    Return
    ------
    bool
        True if extraction pipeline run successfully else False.
    """
    try:
        new_url_run(pdf_url, report_company, str(
            report_year), downloaded=False)
        print("===> EXTRACTION PIPELINE SUCCESS!")
        return True
    except Exception as e:
        print("==> EXTRACTION PIPELINE FAILED!")
        print(e)
        return False


# ============================== DATABASE LOADING ==============================
# Load data on start up
# 1. Load pickle files
loaded_tbl_all = readPickle('assets/data/dashboard_data/tbl_ALL.pickle')

# 2. Load json files
json_input = readData('assets/data/dashboard_data/final_database.json')
json_input = processText(json_input)

# 3. Append each report object in databse for directory list in library page
files = generate_reports()

# ============================== SIDEBAR MAIN CODES ==============================
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
                        [icon_upload,  html.Span("Library")], href="/upload", className="nav-Link")
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        [icon_insight,  html.Span("Insights")], href="/insights"+getCurrReport(), className="nav-Link")
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        [icon_db, html.Span("Dashboard")], href="/dashboard", className="nav-Link")
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

# ============================== DASHBOARD MAIN CODES ==============================


def generate_dashboard(report_name):
    """
    Method to generate dashboard components based on selected report in Library page.

    Parameters
    ----------
    report_name: str
        Company report name selected by user in the library page.

    Return
    ------
    All dashboard dash components.
    """
    dashboard_header = html.Div(
        className="section-header",
        children=[
            html.H6(id="report_name_id", children=[report_name])
        ]
    )
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
        ]
    )
    dashboard_wordcloud1 = html.Div(
        className="section-dashboard_wordcloud1",
        children=[
            html.Img(
                src=app.get_asset_url(
                    report_obj["wordcloud_img_path"][0]
                )
            )
        ]
    )
    dashboard_wordcloud2 = html.Div(
        className="section-dashboard_wordcloud2",
        children=[
            html.Img(
                src=app.get_asset_url(
                    report_obj["wordcloud_img_path"][1]
                )
            )
        ]
    )
    dashboard_wordcloud3 = html.Div(
        className="section-dashboard_wordcloud3",
        children=[
            html.Img(
                src=app.get_asset_url(
                    report_obj["wordcloud_img_path"][2]
                )
            )
        ]
    )

    dashboard_wordcloud4 = html.Div(
        className="section-dashboard_wordcloud4",
        children=[
            html.Img(
                src=app.get_asset_url(
                    report_obj["wordcloud_img_path"][3]
                )
            )
        ]
    )
    dashboard_relevant_tables = html.Div(
        className="section-dashboard_relevant_tables",
        children=[
            html.H6(
                "Decarbonisation-Related Information",
                style={'display': 'inline-block'}
            ),
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
                    },
                    {
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

# ============================== INSIGHTS MAIN CODES ==============================


def getTblKeywords(company, year, pagenumber, index):
    """
    Method to keyword list associated with each data table for each page of a report.

    Parameters
    ----------
    company: str
        Company report name.
    year: int
        Report year.
    pagenumber: int
        Page number of report.
    index: int
        Index for table extracted for each table.

    Return
    ------
    keywords_list : list of str
        List of keywords for each table index in each page.
    """
    keywords_list = []
    for obj in json_input:
        if(obj['company'] == company and obj['year'] == year):
            try:
                keywords_list = obj['table_image_keywords'][pagenumber][index]
            except:
                keywords_list = keywords_list
    return keywords_list


def getChartKeywords(company, year, pagenumber, index):
    """
    Method to keyword list associated with each chart for each page of a report.

    Parameters
    ----------
    company: str
        Company report name.
    year: int
        Report year.
    pagenumber: int
        Page number of report.
    index: int
        Index for table extracted for each table.

    Return
    ------
    keywords_list : list of str
        List of keywords for each chart index in each page.
    """
    keywords_list = []
    for obj in json_input:
        if(obj['company'] == company and obj['year'] == year):
            try:
                keywords_list = obj['chart_images_keywords'][pagenumber][index]
            except:
                keywords_list = keywords_list
    return keywords_list


def changeFilepath(imgpath):
    """
    Method to change image file path due to wrong file path appened in chart extraction pipeline.

    Parameters
    ----------
    imgpath: str
        Image path of chart image.

    Return
    ------
    path : str
        List of keywords for each table index in each page.
    """
    path = imgpath
    if("dashboard_data" not in imgpath):
        path = imgpath.replace("dashboard", "dashboard_data")
    return path


def generate_cards(report_name):
    """
    Method to generate card information for each table and chart to view on insights page.

    Parameters
    ----------
    report_name: str
        Company report name selected by user in the library page.

    Return
    ------
    card_list : dict of {str : str or dict}
        Dictionary containing image paths for each table and charts.
    """
    card_list = []
    count = 0  # counter for images
    report_name = report_name.replace(".pdf", "")
    report_company_name = report_name.split("_")[0]
    report_year = report_name.split("_")[1]

    # Get table images
    for obj in json_input:
        if(obj['company'] == report_company_name and obj['year'] == report_year):
            if(obj['table_images'] == "nan"):
                break
            for page in obj['table_images'].items():
                for index, img in enumerate(page[1]):
                    count = count + 1
                    image_path = {
                        "imagepath": img,
                        "pageNum": page[0],
                        "ext_type": "Table Image",
                        "keywords": getTblKeywords(report_company_name, report_year, page[0], index),
                        "image_count": count,
                    }
                    card_list.append(image_path)
            break

    # Get chart images
    for obj in json_input:
        if(obj['company'] == report_company_name and obj['year'] == report_year):
            if(obj['chart_images'] == "nan"):
                break
            for page in obj['chart_images'].items():
                for index, img in enumerate(page[1]):
                    count = count + 1
                    image_path = {
                        "imagepath": changeFilepath(img),
                        "pageNum": page[0],
                        "ext_type": "Chart Image",
                        "keywords": getChartKeywords(report_company_name, report_year, page[0], index),
                        "image_count": count,
                    }
                    card_list.append(image_path)
            break
    return card_list


def generate_df_cards():
    """
    Method to generate card information for each table dataframe to view on insights page.

    Parameters
    ----------

    Return
    ------
    df_card_lst : dict of {str : str or dict}
        Dictionary containing data points for each table dataframe
    list of str
        List of dropdown values (deprecated)
    """
    df_card_lst = []
    count = 1  # counter for df
    dropdown_vals = ['All', ]

    # Gather table keywords
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

        # Gather dfs
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

    return df_card_lst, sorted(list(set(dropdown_vals)))


def generate_insight(report_name, card_list):
    """
    Method to generate insights components based on selected report in Library page.

    Parameters
    ----------
    report_name: str
        Company report name selected by user in the library page.
    card_list : dict of {str : str or dict}
        Dictionary containing image paths for each table and charts.

    Return
    ------
    All insights dash components.
    """
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
            html.H6(str(len(card_list)) + " Images Found"),

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


# ============================== LIBRARY MAIN CODES ==============================
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
                        for x in reversed(range(1999, getDropdownYears()))
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
# ============================== APP STARTUP CODES ==============================
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
right_content = html.Div(
    id="page-content"
)

app.layout = html.Div(
    className='content', children=[
        dcc.Location(id="url", refresh=True), left_content, right_content
    ]
)
# ============================== ALL BACK-END CALLBACKS ==============================


@ app.callback(
    Output("page-content", "children"),
    [
        Input("url", "pathname")
    ]
)
def render_page_content(pathname):  # Callback for NavBar
    """
    Callback method to route user to selected path.

    Parameters
    ----------
    pathname: str
        Path route name

    Return
    ------
    Dash components based on path route.
    """
    print(f"USER ON PAGE: |{pathname}|")
    # DEFAULT
    if pathname == "/":
        return upload_fn
    # DASHBOARD
    elif pathname == "/dashboard":
        return dcc.Location(pathname="/dashboard/" + getCurrReport(), id="url")
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
        if(getCurrReport() in pathname or len(pathname) > 10):
            pass
        else:
            pathname = pathname + getCurrReport()
        if(pathname != "/insights/"):
            report_name = pathname.split("/")[2]
            setCurrReport(report_name)
            report_curr = getCurrReport()
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


@ app.callback(
    Output("upload-url-output", "children"),
    Input("inputUrl", "value")
)
def url_render(inputUrl):  # Validation for URL link
    """
    Callback method to validate user inputs in upload section in Library page.
    Valid case:
    1. Contains http
    2. Reachable Link
    3. Unique URL link

    Parameters
    ----------
    pathname: str
        Path route name

    Return
    ------
    Error message based on user inputs.
    """
    if (inputUrl is None):
        raise PreventUpdate
    else:
        pass
    # 1. Does not contain http
    if("http" not in inputUrl):
        return u'URL is not a valid link (Does not contain http, or is non-reachable))'
    # 2. Check if link is reachable using request and catch error if link is broken.
    try:
        response = requests.get(inputUrl)
    except:
        return u"URL link is broken or cannot be reached. "
    # 3. Check if pdf link is unique URL link
    for obj in json_input:
        if(obj['url'] == inputUrl):
            return u"Report already exists "
    else:  # valid case
        return u"Valid URL link "


@ app.callback(
    Output("upload-companyname-output", "children"),
    Input("inputCompany", "value")
)
def companyName_render(inputCompany):  # Validation for Company Name
    """
    Callback method to validate user company inputs in upload section in Library page.
    Valid case:
    1. Not empty
    2. >2 characters

    Parameters
    ----------
    inputCompany: str
        Company name input string.

    Return
    ------
    Error message based on user inputs.
    """
    if (inputCompany is None):
        raise PreventUpdate
    else:
        pass
    if(len(inputCompany) < 2):
        return u'Please enter a valid company name, e.g. DBS'
    else:  # valid case
        return


@ app.callback(
    Output('upload-year-output', 'children'),
    [
        Input("inputYear-val", "value")
    ]
)
def update_output(value):  # Validation for Years
    """
    Callback method to validate user company inputs in upload section in Library page.
    Valid case:
    1. Not empty
    2. Selected a year

    Parameters
    ----------
    value: str
        Dropdown year input.

    Return
    ------
    Error message based on user inputs.
    """
    if (value is None):
        raise PreventUpdate
    else:
        pass
    if value is None:
        return "Please select a year."
    else:
        return None


@ app.callback(
    Output("upload-output", "children"),
    Input('submit-val', 'n_clicks'),
    [
        State("inputUrl", "value"),
        State("inputCompany", "value"),
        State("inputYear", "value")
    ],
    prevent_initial_call=True  # Prevent function for running on load
)
# Validation upon clicking submit btn
def input_render(n_clicks, inputUrl, inputCompany, inputYear):
    """
    Callback method to validate user company inputs in upload section in Library page.
    Valid case:
    1. Not empty
    2. Selected a year

    Parameters
    ----------
    n_clicks: str
        Number of clicks on btn.
    inputUrl: str
        Valid pdf url input by user in upload section.
    report_company: str
        Valid report name input by user in upload section.
    report_year: int
        Valid report year input by user in upload section.

    Return
    ------
    Error message based on user inputs or error processing extraction pipeline.
    """
    if n_clicks is None:
        raise PreventUpdate

    # Validate that all fields have been entered
    if (n_clicks > 0) and ((inputUrl is None) or (inputCompany is None) or (inputYear is None)):
        return [html.Ol(className="err-list-ol", children=["- Please enter all fields"])]

    err_message_list = validateInputs(inputUrl, inputCompany, inputYear)
    if(err_message_list == []):
        try:
            new_url, new_company, new_year = inputUrl, inputCompany, inputYear
            print(
                f"====> PREPARING TO ADD NEW REPORT: {new_url}|{new_company}| {new_year}")
            filetoappend = f"{new_company}_{new_year}.pdf"
            if(run_extraction_pipeline(new_url, new_company, new_year) == True):
                global loaded_tbl_all
                global json_input
                global files
                loaded_tbl_all = readPickle(
                    'assets/data/dashboard_data/tbl_ALL.pickle')
                json_input = readData(
                    'assets/data/dashboard_data/final_database.json')
                json_input = processText(json_input)
                files = generate_reports()  # For all files dir
                return dcc.Location(pathname="/insights/"+filetoappend, id="url")
            else:
                return [html.Ol(className="err-list-ol", children=["- URL link cannot be processed."])]
        except Exception as e:
            print(e)
            return [html.Ol(className="err-list-ol", children=["- URL link cannot be processed."])]
    else:
        return [html.Ol(className="err-list-ol", children=[err]) for err in err_message_list]


@ app.callback(
    Output("file-list", "children"),
    Input('searchquery', 'value'),
)
def update_output(input_value):  # Updating file list
    """
    Callback method to search query for reports in Library page.

    Parameters
    ----------
    input_value: str
        Search query for report.

    Return
    ------
    updateList : list of dash components
        List of dash components based on user search input value.
    """
    updateList = generateSearchFileList(input_value)
    return updateList


@ app.callback(
    Output('table-container', 'data'),
    [
        Input('filter_dropdown', 'value')
    ]
)
def display_table(state):
    """
    (Deprecated) Display dataframe tables in Insights page.

    Parameters
    ----------
    state: str
        Dropdown category selection.

    Return
    ------
    updateList : Dict of table records
        Dictionary of table data.
    """
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

# For formatting large json file
# cat final_database.json | python -m json.tool > finaldb.json
