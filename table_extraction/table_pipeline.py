# import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileReader
import requests, io
import gzip
import requests
import pdf2image
import cv2
import camelot
import pandas as pd

# import Multi-Type-TD-TSR
import torch, torchvision
import pytesseract
import detectron2
import Multi_Type_TD_TSR.google_colab.deskew as deskew
import Multi_Type_TD_TSR.google_colab.table_detection as table_detection
import Multi_Type_TD_TSR.google_colab.table_xml as txml
import Multi_Type_TD_TSR.google_colab.table_ocr as tocr
import pandas as pd
import os
import json
import itertools
import random
from detectron2.utils.logger import setup_logger

# import common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
setup_logger()

#create detectron config
cfg = get_cfg()

#set yaml
cfg.merge_from_file('All_X152.yaml')

#set model weights
cfg.MODEL.WEIGHTS = 'model_final.pth' # Set path model .pth

predictor = DefaultPredictor(cfg) 

# list of keywords
ESG_DICTIONARY = ['ghg', 'scope 1', 
                'scope 2', 'scope 3', 'energy', 
                'paper', 'green bonds', 'renewable energy',
                'water', 'carbon intensity', 
                'carbon emissions', 'waste', 
                'electricity', 
                'weighted average carbon intensity', 'WACI']


# list of units
ESG_UNITS = ['tonnes', 'tons', 'kWh', 'kilogram', 'kilowatt hour', 
           'gigajoules', 'GJ', 'litre', 'liter', 'CO2e', 'tCO', 't CO', 'MWh', 
           'megawatt hour', 'GWh', 'gigawatt hour',
           'cubic metres', 'cm3', 'm3', 'per employee', 'ream', 'quire', 'sheet', 'bundle', 'bale']

def isoverlap(r1, r2):
    """
    Check if two extracted table images have overlapping coordinates.

    Parameters
    ----------
    r1 : list of int
        [x, y, w, h] coordinates of 1st rectangle.
    r2 : list of int
        [x, y, w, h] coordinates of 2nd rectangle.

    Return
    ------
    bool
        True if the coordinates overlap else False.
    """
    y1 = r1[1]
    x1 = r1[0]
    h1 = r1[3]
    w1 = r1[2]
    
    y2 = r2[1]
    x2 = r2[0]
    h2 = r2[3]
    w2 = r2[2]
    
    if ((x1+w1)<x2 or (x2+w2)<x1 or (y1+h1)<y2 or (y2+h2)<y1):
        return False
    else:
        return True
    
def check_coord_overlap(table_coords):
    """
    Returns new list of non-overlapping table coordinates.

    Parameters
    ----------
    table_coords : list of list of int
        List of [x, y, w, h] coordinates of rectangles (rectangular boundaries of table images).

    Return
    ------
    new_table_coords : list of list of int
        List of [x, y, w, h] coordinates of non-overlapping rectanglular table images.
    """
    new_table_coords = []
    
    for i in range (len(table_coords)):
        flag = True
        if i == 0:
            new_table_coords.append(table_coords[i])
        else:
            for j in range (len(new_table_coords)):
                if isoverlap(table_coords[i], new_table_coords[j]):
                    flag = False
                    break
            if flag:
                new_table_coords.append(table_coords[i])
    
    return new_table_coords

def get_pdf_dimension(pdf_url):
    """
    Retrieve dimensions of PDF page.

    Parameters
    ----------
    pdf_url : str
        URL of PDF report.

    Return
    ------
    height : float
        Height of PDF page.
    width : float
        Width of PDF page.
    """
    response = requests.get(pdf_url)
    with io.BytesIO(response.content) as open_pdf_file:
        pdf = PdfFileReader(open_pdf_file, strict=False)
        height = pdf.getPage(0).mediaBox.getHeight()
        width = pdf.getPage(0).mediaBox.getWidth()
    return height, width

def get_image_and_dimension(pdf_url):
    """
    Convert PDF to PIL image and get dimensions of image.

    Parameters
    ----------
    pdf_url : str
        URL of PDF report.

    Return
    ------
    height : float
        Height of PDF page.
    width : float
        Width of PDF page.
    images : list of objects
        List of PIL images converted from all pages in PDF report.
    """
    response = requests.get(pdf_url, stream=True, timeout=30)
    # pdf = gzip.open(response.raw)
    # images = pdf2image.convert_from_bytes(pdf.read())
    images = pdf2image.convert_from_bytes(response.content, size=1000)
    pg_1_img = images[0] # type=PIL.PpmImagePlugin.PpmImageFile
    width, height = pg_1_img.size
    return height, width, images

def get_scaling_factor(pdf_height, pdf_width, img_height, img_width):
    """
    Get scaling factor of extracted table images.

    Parameters
    ----------
    pdf_height : float
        Height of PDF report.
    pdf_width : float
        Width of PDF report.
    img_height : float
        Height of extracted image.
    img_width : float
        Width of extracted image.

    Return
    ------
    scaling_factor_height : float
        Scaling factor for height.
    scaling_factor_width : float
        Scaling factor for width.
    """
    scaling_factor_height = img_height/pdf_height
    scaling_factor_width = img_width/pdf_width
    return scaling_factor_height, scaling_factor_width

def convert_PIL_cv2(pdf_pil_img):
    """
    Convert PIL images to CV images.

    Parameters
    ----------
    pdf_pil_img : list of objects
        List of PIL images.

    Return
    ------
    pdf_cv2_img : list of objects
        List of CV images.
    """    
    pdf_cv2_img = []
    for pil_img in pdf_pil_img:
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        pdf_cv2_img.append(cv2_img)
    return pdf_cv2_img

def isfloat(value):
    """
    Check if value is numerical.

    Parameters
    ----------
    value : any

    Return
    ------
    bool
        True if value is of type float else False.
    """    
    try:
        float(value)
        return True
    except ValueError:
        return False
    
def num_there(s):
    """
    Check if any character is a digit.

    Parameters
    ----------
    s : str

    Return
    ------
    bool
        True if character is digit else False.
    """  
    return any(i.isdigit() for i in s)

# list of units to be used for filtering - note: this is different from the ones under Table Detection section! 
UNITS = ['tonnes', 'tons', 'kWh', 'kilogram', 'kilowatt hour', 
           'gigajoules', 'GJ', 'litre', 'liter', 'CO2e', 'tCO', 't CO', 'MWh', 
           'megawatt hour', 'GWh', 'gigawatt hour',
           'cubic metres', 'cm3', 'm3', 'per employee', 'ream', 'quire', 'sheet', 'bundle', 'bale', '%', 't']

def clean_tbl(tbl):
    """
    Perform cleaning of dataframe.

    Parameters
    ----------
    tbl : df

    Return
    ------
    tbl : df or None
        Cleaned df if df can be cleaned else None.
    final_words : list of str or None
        List of ESG keywords if keyword found in dataframe else None.
    """  
    # replace subscript and newline 
    tbl = tbl.replace(r'(<s>).*(</s>)','',regex=True)
    tbl = tbl.replace(r'\n','',regex=True)
    # convert 1st row to header 
    header_df = tbl.iloc[0] #grab the first row for the header
    tbl = tbl[1:]
    tbl.columns = header_df 
    # remove comma in numeric values 
    tbl = tbl.apply(lambda x: x.str.replace(',',''))
    # remove brackets surrounding numeric metrics 
    tbl = tbl.replace(r"\((\d+)\)", r"\1", regex=True)
    # loop through each cell and check if they are float/num or they are metrics with units 
    for row in range(tbl.shape[0]):
        for col in range(1, tbl.shape[1]):
            value = tbl.iloc[row, col]
            if len(value.split()) > 3:
                tbl.iloc[row,col] = np.nan
            elif isfloat(value) or (any(substring in value for substring in UNITS) and num_there(value)):
                continue 
            else:
                tbl.iloc[row,col] = np.nan
    # drop columns with > 80% NaN
    tbl = tbl.loc[:, tbl.isnull().mean() < .8]
    # drop rows with any NaN
    tbl = tbl.dropna()
    if (tbl.shape[1] == 1) or (tbl.shape[0] == 0): # if there's only 1 col left or 0 row left 
        return None, None 
    page_kw = ['page', 'Page', 'PAGE']
    for s in page_kw:
        if any(s in h for h in tbl.columns):
            return None, None 
    first_column = tbl.iloc[:, 0] # get first column of tbl 
    num_of_nan = first_column.isnull().sum(axis = 0)
    # large proportion of nan cells in 1st column
    if num_of_nan/len(first_column) > 0.8:
        return None, None
    # no headers 
    headers =tbl.columns
    if not(any(h for h in headers)):
        return None, None 
    # list of words in df for relevance 
    words = pd.unique(tbl.values.ravel())
    words = pd.unique([word for line in words for word in line.split()])
    final_words = []
    for s in ESG_DICTIONARY:
        if any(s in word.lower() for word in words):
            final_words.append(s) 
    for s in ESG_DICTIONARY:
        if any(s in word.lower() for word in tbl.columns):
            final_words.append(s)
    final_words = list(set(final_words))
    return tbl, final_words 

def extract_tbl_from_page(page_idx, pdf_cv2_img, scaling_factor_height, scaling_factor_width, img_height, url, path):
    """
    Extract text-based and image-based table from individual page of PDF report.

    Parameters
    ----------
    page_idx : int
        Page index of PDF report
    pdf_cv2_img : list of objects
        List of CV images converted from PDF pages.
    scaling_factor_height : float
        Scaling factor for height.
    scaling_factor_width : float
        Scaling factor for width.
    img_height : float
        Original image height.
    url : str
        URL of PDF report.
    path : str
        Path directory where extracted table images are saved.

    Return
    ------
    tbl_lst : list of df
        List of cleaned dataframes.
    tbl_keywords_lst : list of list of str
        List of keywords lists identified from each extracted dataframe in a PDF page.
    img_keywords_lst : list of list of str
        List of keywords lists identified from each extracted table image in a PDF page.
    lst_imgs: list of str
        List of image paths for every detected table image in a PDF page.
    """  
    table_list, table_coords = table_detection.make_prediction(pdf_cv2_img[page_idx], predictor)
    tbl_lst = []
    tbl_keywords_lst = []
    img_keywords_lst = []
    lst_imgs = []
    
    table_coords = check_coord_overlap(table_coords)
    
    count = -1
    # ---- start Table extraction ---- /recount from every page 
    i = 0 
    # ---- end Table extraction ----
    for table_coord in table_coords:
        count += 1
        try: 
            x1 = table_coord[0]
            y1 = table_coord[1]
            x2 = table_coord[2] + table_coord[0]
            y2 = table_coord[3] + table_coord[1]
            
            # extract detected table by coordinates
            page_img = pdf_cv2_img[page_idx]
            page_img = page_img[y1:y2, x1:x2]
            scale_percent = 150 # percent of original size
            width = int(page_img.shape[1] * scale_percent / 100)
            height = int(page_img.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            resized = cv2.resize(page_img, dim)
            #cv2_imshow(resized)
            text = pytesseract.image_to_string(resized)
            
            # if any keyword found in text in table, save image into output folder
            if any(keyword in text.lower() for keyword in ESG_DICTIONARY) and any(unit in text for unit in ESG_UNITS):
                print("----------------------------------------------------------------------------------------------------------")
                print("RELEVANT IMAGE FOUND!, tbl_num is {0}".format(count))
                print("----------------------------------------------------------------------------------------------------------")
                
                # cv2 and pdf used 2 different coordinate systems, thus y1 and y2 need to be modified
                y1 = img_height - y1
                y2 = img_height - y2
                
                # scale by scaling factors
                scaled_x1 = x1/scaling_factor_width
                scaled_x2 = x2/scaling_factor_width
                scaled_y1 = y1/scaling_factor_height
                scaled_y2 = y2/scaling_factor_height
                edge_tol = scaled_x2 - scaled_x1
                coords = f"{scaled_x1}, {scaled_y1}, {scaled_x2}, {scaled_y2}"
                
                tbls_scaled = camelot.read_pdf(url, flavor='stream', edge_tol=edge_tol, pages=str(page_idx+1), flag_size=True, table_areas=[coords], split_text=True) # split_text=True
                tbl_cleaned, ESG_keywords = clean_tbl(tbls_scaled[0].df)
                # there is a valid table 
                
                if tbl_cleaned is None:
                    continue
                else:
                    if len(ESG_keywords) > 0:
                        print("----------------------------------------------------------------------------------------------------------")
                        print("CLEANED TABLE OBTAINED!, tbl_num is {0}".format(count))
                        print("----------------------------------------------------------------------------------------------------------")
                        tbl_lst.append(tbl_cleaned)
                        tbl_keywords_lst.append(ESG_keywords)
                        img_keywords_lst.append(ESG_keywords)
                        # Table detection portion saving 
                        # --- start ---
                        output_path = f"{path}/PAGE{str(page_idx+1)}_IMAGE{str(i)}.jpg"
                        cv2.imwrite(output_path, page_img)
                        i += 1
                        lst_imgs.append(output_path)
                        # --- end ---
                    else:
                        print("----------------------------------------------------------------------------------------------------------")
                        print("SAVING ONLY IMAGE!, img_num is {0}".format(count))
                        print("----------------------------------------------------------------------------------------------------------")
                        # --- start ---
                        output_path = f"{path}/PAGE{str(page_idx+1)}_IMAGE{str(i)}.jpg"
                        cv2.imwrite(output_path, page_img)
                        i += 1
                        lst_imgs.append(output_path)
                        # --- end ---
                        kw_lst = []
                        for t in ESG_DICTIONARY:
                            if t in text.lower():
                                kw_lst.append(t)
                        img_keywords_lst.append(kw_lst)
                        
                        
        except Exception as e:
            print(e)
            print('error on page {0}'.format(page_idx+1))
            continue 
    
    return tbl_lst, tbl_keywords_lst, img_keywords_lst, lst_imgs

def extract_tbl_from_pdf(pdf_url, pages_to_look_for, path):
    """
    Create dictionaries in the form of {page no.: output}.

    Parameters
    ----------
    pdf_url : str
        URL of PDF report.
    pages_to_look_for : list of str
        List of relevant page numbers of PDF report.
    path : str
        Path directory where extracted table images are saved.

    Return
    ------
    pdf_dict : dict of {str : list of df}
        Dictionary of {page: list of cleaned dataframes}.
    tbl_keywords_dict : dict of {str : list of list}
        Dictionary of {page: list of keywords lists identified from each extracted dataframe}.
    img_keywords_dict : dict of {str : list of list}
        Dictionary of {page: list of keywords lists identified from each extracted table image}.
    image_path_obj: dict of {str : list of str}
        Dictionary of {page: list of image paths for every detected table image}.
    """  
    pdf_dict = {}
    tbl_keywords_dict = {}
    img_keywords_dict = {}
    image_path_obj = {}
    
    # consistent to all pages in a pdf_url 
    print('getting pdf & img height, width...')
    pdf_height, pdf_width = get_pdf_dimension(pdf_url) # original pdf height,width 
    img_height, img_width, pdf_pil_img = get_image_and_dimension(pdf_url) # original image height,width 
    scaling_factor_height, scaling_factor_width = get_scaling_factor(pdf_height, pdf_width, img_height, img_width)
    print('retrieved pdf & img height, width')
    
    # convert PIL images to CV images
    print('converting all pages to CV2 img...')
    pdf_cv2_img = convert_PIL_cv2(pdf_pil_img)
    print ('converted all pages to CV2 img')
    
    # pdf_cv2_img = images of all pages in a pdf 
    print('looping through each page...')
    for page_no in pages_to_look_for:
        print('retrieving from page {0}'.format(page_no))
        page_idx = int(page_no) - 1
        tbl_lst, tbl_keywords_lst, img_keywords_lst, img_lst= extract_tbl_from_page(page_idx, pdf_cv2_img, scaling_factor_height, scaling_factor_width, img_height, pdf_url, path)
        pdf_dict[page_no] = tbl_lst
        tbl_keywords_dict[page_no] = tbl_keywords_lst
        image_path_obj[page_no] = img_lst
        img_keywords_dict[page_no] = img_keywords_lst
        
    
    return pdf_dict, tbl_keywords_dict, img_keywords_dict, image_path_obj

def table_pipeline(report):
    """
    Main table pipeline function.

    Parameters
    ----------
    report : dict of {str : str or dict}
        Dictionary of a company's report details and preprocessed text.

    Return
    ------
    pickle : dict of {str : str or dict}
        Dictionary containing a company's name, year, PDF URL and cleaned dataframes.
    report : dict of {str : str or dict}
        Dictionary containing a company's report details, preprocessed text and table pipeline output.
    """          
    # basic information
    company = report['company']
    year = report['year']
    pdf_url = report['url']

    # create dictionary for report 
    pickle = {}
    pickle['company'] = company
    pickle['year'] = year
    pickle['url'] = pdf_url

    path = 'data/dashboard_data/table_images/' + company + '_' + year
    os.mkdir(path)
    print(f"Detection for Report: {company}_{year}")

    # relevant pages for ESG info extraction
    pages_to_look_for = []
    for page in report['filtered_report_tables_direct']:
        pages_to_look_for.append(page)

    for page in report['filtered_report_tables_indirect']:
        pages_to_look_for.append(page)

    try:
        print('calling extract_tbl_from_pdf')
        pickle['tbl_pages'], report['table_keywords'], report['table_image_keywords'], report['table_images']= extract_tbl_from_pdf(pdf_url, pages_to_look_for, path) # returns a dict  
        print('Successful for {0}, {1}'.format(company, year))
    except Exception as e:
        print(e)
        print("Error occurred in table_extraction/ Request failed")
        pickle['tbl_pages'] = []
        report['table_keywords'] = 'nan'
        report['table_image_keywords'] = 'nan'
        report['table_images'] = 'nan'

    return pickle, report


# FOR TESTING PURPOSES - @XinMin can call the table_pipeline(company_dict) function directly in the main pipeline
file_path = 'all_asset_managers_preprocessed_vfinal.json'
f = open(file_path,)
data = json.load(f)
pickle, report = table_pipeline(data[1])
