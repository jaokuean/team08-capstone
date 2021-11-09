# Use for checking files in dir
import os

# Extract each pdf page to image
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

# Image Processing
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pytesseract
import json
import requests
from pathlib import Path
import re
import datetime
import shutil  # high-level folder management library

# Optimal parameters for graph detection
set_column_gap = 50
set_height_limit = 180
set_width_limit = 180
set_area_limit = 100000

set_scale_factor = 3.2  # Extend horizontal and vertical axis of bounding boxes
scale_horizontal = set_scale_factor*64
scale_vertical = set_scale_factor*64


def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

# Method to check if extracted article is valid


def checkDim(height, area):
    """
    Primary filter using heuristic approach
    ----------
    height: str
        Height of pdf page converted from pytesseract
    area: str
        Area of pdf page converted from pytesseract

    Return
    ------
    True/False : boolean
        If image pass filter rules = True else False
    """
    #print(f"{height} x {width} = {area}")
    if(height <= set_height_limit):
        return False
    if(area <= set_area_limit):
        return False
    return True


def process_Num(text):
    """
    Process text to return cleaned digits.
    ----------
    text: str
        Text of pdf page converted from pytesseract

    Return
    ------
    results : list
        List of digits found after removing punctuations and string values
    """
    no_punct = re.sub('[^0-9\n\.]', ' ', text)
    res = no_punct.split()
    my_list = set(res)
    to_delete = ["."]

    my_list.difference_update(to_delete)
    results = list(my_list)
    return results


def process_text(text):
    """
    Process text to return cleaned strings
    ----------
    text: str
        Text of pdf page converted from pytesseract

    Return
    ------
    results : list
        List of string found after removing punctuations
    """
    no_punct = re.sub('[^a-zA-Z\n\.]', ' ', text)
    res = no_punct.split()
    my_list = set(res)
    to_delete = ["."]

    my_list.difference_update(to_delete)
    results = list(my_list)
    return results


def find_keywords(text):
    """
    Get total number of unique keywords found after removing punctuations.
    ----------
    text: str
        Text of pdf page converted from pytesseract

    Return
    ------
    results : int
        List of unique keywords found after removing punctuations
    value : int
        Total number of unique keywords found after removing punctuations
    """
    no_punct = re.sub('[^a-zA-Z0-9\n\.]', ' ', text)
    # list of keywords
    dictionary = {'carbon', 'ghg', 'emission',
                  'emissions', "scope", "WACI", "net-zero",
                  'energy', 'water', 'waste', 'coal', 'power', 'green', 'paper', 'consumption', 'renewable',
                  'breakdown', 'loans', 'tonnes', 'tons', 'kWh', 'kg', 'kilogram', 'kilowatt hour',
                  'gigajoules', 'GJ', 'litre', 'liter', 'CO2e', 'tCO', 't CO', 'MWh',
                  'megawatt hour', '%', 'cubic metres', 'per employee', 'm3', 'co2', 'o2', 'million', 'total', 'trillion', 'set'
                  }
    res = set(no_punct.lower().split())
    print(res)
    newlength = len(res)
    res.intersection_update(dictionary)
    print(res)
    results = list(res)
    value = len(results)
    return results, value


def count_clean_text(text):
    """
    Get total number of text after removing punctuations and digits 
    ----------
    text: str
        Text of pdf page converted from pytesseract

    Return
    ------
    len(res) : int
        Total number of text after removing punctuations
    """
    no_punct = re.sub('[^a-zA-Z\n\.]', ' ', text)
    res = no_punct.split()
    return len(res)


def count_clean_num(text):
    """
    Get total number of digits after removing punctuations and string values
    ----------
    text: str
        Text of pdf page converted from pytesseract

    Return
    ------
    len(res) : int
        Total number of digits after removing punctuations
    """
    no_punct = re.sub('[^0-9\n\.]', ' ', text)
    res = no_punct.split()
    return len(res)


def getTotalLen(text):
    """
    Get total length of string value after removing punctuations 
    ----------
    text: str
        Text of pdf page converted from pytesseract

    Return
    ------
    len(res) : int
        Total number of string value after removing punctuations
    """
    no_punct = re.sub('[^a-zA-Z0-9\n\.]', ' ', text)  # remove punctuations
    res = no_punct.split()
    return len(res)


def filter_relevance(filter_img):
    """
    Filter unwanted image using heuristic approach and rules based on text and relevance
    Parameters
    ----------
    filter_img: image
        Image of pdf page converted from pdf2image

    Return
    ------
    True/False : boolean
        If image pass filter rules = True else False
    """
    try:
        text = pytesseract.image_to_string(filter_img)
    except:
        print("ERROR at Textserract")
        return

    # Filter images with too much text
    # Total length of text
    total_len = getTotalLen(text)

    # Filter images with too little or no keywords
    # Total unique keywords found
    listKeywords, keywords = find_keywords(text)

    # Get total unique string
    textonly_len = len(process_text(text))

    # Get total unique numbers/digits
    numonly_len = len(process_Num(text))

    # Get text to total text ratio = clean_text/total text
    try:
        tt_ratio = count_clean_text(text)/total_len
    except:
        tt_ratio = 0

    # Basic shape descriptive data
    height = filter_img.shape[0]
    width = filter_img.shape[1]
    # number of components used to represent each pixel.
    channels = filter_img.shape[2]
    area = height * width

    # num to area ratio = total num to whole image
    na_ratio = count_clean_num(text)/area * 10**5

    # Color processing
    # Filter by BW instead of colors, cause too similar
    gray = cv2.cvtColor(filter_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    # Count total white and black ratio
    white_pix = np.sum(dilate == 255)
    black_pix = np.sum(dilate == 0)
    # b/w ratio
    bw_ratio = black_pix/white_pix

    if(keywords > 0 and tt_ratio < 0.99 and bw_ratio > 14 and textonly_len < 90 and numonly_len > 1 and na_ratio > 0.45):
        return True
    else:
        return False


def filterImage(img):
    """
    Filter unwanted image using heuristic approach and rules based on image properties
    Parameters
    ----------
    img: image
        Image of pdf page converted from pdf2image

    Return
    ------
    True/False : boolean
        If image pass filter rules = True else False
    """
    try:
        text = pytesseract.image_to_string(img)
    except:
        print("ERROR at Textserract")
        return

    # Filter images with too much text
    # Total length of text
    total_len = getTotalLen(text)

    # Basic shape descriptive data
    height = img.shape[0]
    width = img.shape[1]
    # number of components used to represent each pixel.
    channels = img.shape[2]
    area = height * width
    # text to area ratio = total text to whole image
    ta_ratio = count_clean_text(text)/area * 10**5

    # Color processing
    # Filter by BW instead of colors, cause too similar
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    dilate_raw = cv2.mean(dilate)[::-1]
    dilated_region = int(dilate_raw[3])

    # Count total white pixels
    white_pix = np.sum(dilate == 255)

    # region dilated + area captured
    if((7 < dilated_region < 26) and (7000 < white_pix < 90000) and total_len < 68 and ta_ratio < 10):
        return True
    else:
        return False


def scale_image(x_top, y_top, x_bot, y_bot, result):
    """
    Scale image for processing.
    Parameters
    ----------
    x_top: int
        Top left corner of image
    y_top: int
        Top right corner of image
    x_bot: int
        Bottom left corner of image
    y_bot: int
        Bottom right corner of image
    result: image
        Image to scale

    Return
    ------
    output : int
        Scaled image
    """
    # Extend bounding lines
    new_x_top = x_top - int(scale_horizontal)
    new_y_top = y_top - int(scale_vertical)
    new_x_bot = x_bot + int(scale_horizontal)
    new_y_bot = y_bot + int(scale_vertical)

    # To prevent error on -ve values
    if(new_x_top < 0):
        new_x_top = 0
    if(new_y_top < 0):
        new_y_top = 0
    if(new_x_bot < 0):
        new_x_bot = 0
    if(new_y_bot < 0):
        new_y_bot = 0

    # Re-calculate bounding lines
    new_width = new_x_bot - new_x_top
    new_height = new_y_bot - new_y_top

    X, Y, W, H = new_x_top, new_y_top, new_width, new_height

    output = result[Y:Y+H, X:X+W]

    return output


def process_image(img, pageNum, task, keyword_list, img_list, fileHeader):
    """
    Process each image and find charts in a page.
    Parameters
    ----------
    img: image
        Image of pdf page converted from pdf2image
    pageNum: int
        Relevant page number of raw pdf
    task: str
        If task is a/b, its part a and b of lanscape images else normal portrait page
    img_list: int
        Store image path of ROI images
    fileHeader: int
        Path name of output folder

    Return
    ------
    img_list : int
        List of company's report chart ROI image path for each relevant page.
    keyword_list : int
        List of keywords corresponding to each relevant page.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # turn img to grey
    img_gray_inverted = 255 - img_gray  # Invert back to normal

    row_means = cv2.reduce(img_gray_inverted, 1,
                           cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
    row_gaps = zero_runs(row_means)
    row_cutpoints = (row_gaps[:, 0] + row_gaps[:, 1] - 1) / 2

    bounding_boxes = []
    for n, (start, end) in enumerate(zip(row_cutpoints, row_cutpoints[1:])):
        line = img[int(start):int(end)]
        line_gray_inverted = img_gray_inverted[int(start):int(end)]

        column_means = cv2.reduce(
            line_gray_inverted, 0, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
        column_gaps = zero_runs(column_means)
        column_gap_sizes = column_gaps[:, 1] - column_gaps[:, 0]
        column_cutpoints = (column_gaps[:, 0] + column_gaps[:, 1] - 1) / 2

        filtered_cutpoints = column_cutpoints[column_gap_sizes >
                                              set_column_gap]

        for xstart, xend in zip(filtered_cutpoints, filtered_cutpoints[1:]):
            bounding_boxes.append(
                ((int(xstart), int(start)), (int(xend), int(end))))

    count = 0
    result = img.copy()

    for bounding_box in bounding_boxes:
        count = count + 1  # count number of images extracted

        x_top = bounding_box[0][0]
        y_top = bounding_box[0][1]
        x_bot = bounding_box[1][0]
        y_bot = bounding_box[1][1]

        height = y_bot - y_top  # height of image extracted
        width = x_bot - x_top  # width of image extracted
        area = height * width  # area of image extracted

        X, Y, W, H = x_top, y_top, width, height
        # Primary Filtering
        if(checkDim(height, area) == True):
            print(
                f"============ PASSED for AREA FILTER {pageNum}_{task}_{count}===================")
            # if condition pass, crop image and output

            cropped_image = result[Y:Y+H, X:X+W]

            article_output_file = fileHeader + "/filter_1/page" + \
                str(pageNum) + "_"+str(task)+"_" + str(count) + ".png"

            cv2.imwrite(article_output_file, cropped_image)

            # For data collection
            # article_output_file_2 = "ChartExtraction_Output/out/filter_1/page" + str(pageNum) + "_"+str(task)+"_" +str(count) +".png"
            # cv2.imwrite(article_output_file_2, cropped_image)

            # Black-White Filtering
            filter_img = cv2.imread(article_output_file, cv2.IMREAD_COLOR)
            if filterImage(filter_img) == True:
                scale_image_size = scale_image(
                    x_top, y_top, x_bot, y_bot, result)

                article_output_file = fileHeader + "/filter_2/page" + \
                    str(pageNum) + "_"+str(task)+"_" + str(count) + ".png"
                try:
                    cv2.imwrite(article_output_file, scale_image_size)
                except:
                    print("ERROR at Black-White Filter")
                    print(scale_image_size)
                filter_img = cv2.imread(article_output_file, cv2.IMREAD_COLOR)
                # Relevance Filtering
                if filter_relevance(filter_img) == True:
                    print(
                        f"============ ACCEPTED {pageNum}_{task}_{count}===================")
                    ROI_image_path = fileHeader + "/ROI_" + \
                        str(pageNum) + "_" + str(task) + \
                        "_" + str(count) + ".png"
                    try:
                        text = pytesseract.image_to_string(filter_img)
                    except:
                        print("ERROR at Textserract")
                        return
                    listKeywords, keywords = find_keywords(text)
                    keyword_list.append(listKeywords)
                    cv2.imwrite(ROI_image_path, filter_img)
                    img_list.append(ROI_image_path)
                else:
                    print(
                        f"============ REJECTED at RELEVANCE FILTER {pageNum}_{task}_{count}===================")
            else:
                print(
                    f"============ REJECTED for COLOR FILTER {pageNum}_{task}_{count}===================")
        else:
            print(
                f"============ REJECTED for AREA FILTER {pageNum}_{task}_{count}===================")

    return img_list, keyword_list


def isLandscape(h, w):
    """
    Check if a pdf page is portrait or landscape
    Parameters
    ----------
    h: int
        Height of image
    w: int
        Width of image
    Return
    ------
    True/False : Boolean
        True = landscape, False = portrait
    """
    if(w > h):
        return True
    else:
        return False


def page_to_articles(pageNum, fileHeader):
    """
    Read relevant page and save all images ROI in a list.
    Parameters
    ----------
    pageNum: int
        Relevant page number of raw pdf
    fileHeader: int
        Path name of output folder
    Return
    ------
    img_list : int
        List of company's report chart ROI image path for each relevant page.
    keyword_list : int
        List of keywords corresponding to each relevant page.
    """
    print(f"=>starting on page {pageNum} : {fileHeader}")
    img_list = []
    keyword_list = []
    img = cv2.imread(fileHeader+'/pages/%s.png' %
                     pageNum, cv2.IMREAD_COLOR)  # Identify img
    h, w, c = img.shape

    # Check if image is portrait or landscape
    if(isLandscape(h, w) == True):
        width_cutoff = w // 2
        s1 = img[:, :width_cutoff]
        s2 = img[:, width_cutoff:]
        process_image(s1, pageNum, "a", keyword_list, img_list, fileHeader)
        process_image(s2, pageNum, "b", keyword_list, img_list, fileHeader)
    else:
        process_image(img, pageNum, "0", keyword_list, img_list, fileHeader)

    return keyword_list, img_list


def chart_extraction(url, pages, copy_to_path):
    """
    Execute chart/graph pipeline function.
    Parameters
    ----------
    out_folder: dict of {str : str or dict}
        Path name of output folder for processing
    Return
    ------
    image_path_obj : int
        Dictionary of all company's report chart ROI image path for each relevant page.
    image_keywords_path_obj : int
        Dictionary of keywords corresponding to each relevant page.
    """
    # check if URL is pdf
    if ".pdf" not in url:
        print("URL is not a PDF.")
        return "nan"

    try:
        response = requests.get(url)
    except:
        print("Requests failed.")
        return "nan"

    i = 1
    image_path_obj = {}
    image_keywords_path_obj = {}

    # Convert relevant pages to images for processing
    response = requests.get(url, timeout=30)
    images = convert_from_bytes(response.content)

    for i, image in enumerate(images):
        target_file = copy_to_path+"/pages"  # Create dir for page output
        if not os.path.exists(target_file):
            # file exists
            os.mkdir(target_file)

        if str(i) not in pages:
            continue
        print(f"==> Convert page {i} of pdf to image...")
        image.save(f"{target_file}/{str(i)}.png")

    # Create dir for chart extraction output
    target_file_a = copy_to_path+"/filter_1"
    if not os.path.exists(target_file_a):
        # file does not exists
        os.makedirs(target_file_a)

    target_file_b = copy_to_path+"/filter_2"  # Create dir for test outputs
    if not os.path.exists(target_file_b):
        # file does not exists
        os.makedirs(target_file_b)
    try:
        for page in pages:
            print(f"==> Now doing page {page} ...")
            keyword_list, img_list = page_to_articles(page, copy_to_path)
            print(f"==> Finished Page {page} ...\n")
            image_path_obj[str(page)] = img_list
            image_keywords_path_obj[str(page)] = keyword_list
    except Exception as e:
        print(f"ERROR at page_to_article: ")
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        image_path_obj[str(page)] = "nan"
        image_keywords_path_obj[str(page)] = "nan"

    return image_keywords_path_obj, image_path_obj


def rm_processing_folders(out_folder):
    """
    Remove processing folders to run chart/graph pipeline function.
    Parameters
    ----------
    out_folder: str
        Path name of output folder for processing
    Return
    ------
    total_ROI : int
        Total number of ROI from company's report chart pipeline output.
    """
    total_ROI = 0
    if not os.path.exists(out_folder):
        print("===> Folder do not exist")
    else:
        hidden_file = out_folder+"/.DS_Store"
        if os.path.exists(hidden_file):
            os.remove(hidden_file)

        removed_folders = []
        for filename in os.listdir(out_folder):
            print(f"[Cleaning {filename}]")
            img_path = os.path.join(out_folder, filename)
            removed_folders = []
            for sub_filename in os.listdir(img_path):
                if sub_filename.endswith("filter_1") or sub_filename.endswith("filter_2") or sub_filename.endswith("pages"):
                    delete_path = os.path.join(img_path, sub_filename)
                    # print(f" ====>{delete_path}")
                    removed_folders.append(delete_path)
                    shutil.rmtree(delete_path)

            # Calculate total ROI
            roi_count = len(os.listdir(os.path.join(out_folder, filename)))
            total_ROI = total_ROI + roi_count

            print(f"=> ROI count: {roi_count} ")
            print(f"=> Succesfuly removed: {removed_folders}\n")

    return total_ROI


def run_extraction_main(json_data, out_folder):
    """
    Process json data and pass into chart/graph pipeline function.
    Parameters
    ----------
    json_data : dict of {str : str or dict}
        Dictionary of a company's report details and preprocessed text.
    out_folder: str
        Path name of output folder for processing
    Return
    ------
    image_path_obj : int
        Dictionary of all company's report chart ROI image path for each relevant page.
    image_keywords_path_obj : int
        Dictionary of keywords corresponding to each relevant page.
    """
    company = json_data['company']
    year = json_data['year']
    pdf_url = json_data['url']

    pages = []
    for j in json_data['filtered_report_tables_direct']:
        pages.append(j)

    path = out_folder + company + '_' + year

    if not os.path.exists(path):
        # file exists
        os.mkdir(path)

    json_obj = {}
    image_path_obj = {}
    image_keywords_path_obj = {}
    try:
        image_keywords_path_obj, image_path_obj = chart_extraction(pdf_url, pages, path)
        json_data['chart_images_keywords'] = image_keywords_path_obj
        json_data['chart_images'] = image_path_obj
        return json_data
    except Exception as e:
        print(f"ERROR at chart_extraction: ")
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        json_data['chart_images_keywords'] = "nan"
        json_data['chart_images'] = "nan"
        return json_data


def chart_pipeline(data):
    """
    Main chart/graph pipeline function.
    Parameters
    ----------
    data : dict of {str : str or dict}
        Dictionary of a company's report details and preprocessed text.

    Return
    ------
    data : dict of {str : str or dict}
        Dictionary containing a company's report details, preprocessed text and table pipeline output.
    """
    # TODO: place to store chart output
    out_folder = "data/dashboard/ChartExtraction_Output/"

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    data = run_extraction_main(data, out_folder)


    # Delete processing folders
    # out_folder = "ChartExtraction_Output"
    try:
        ROI_count = rm_processing_folders(out_folder)
        print(
            f"[Summary Report]\n Time Completed: {datetime.datetime.now}  \nTotal ROI: {ROI_count}")
    except Exception as e:
        print(f"Fail to remove: {e}")

    return data

# FOR TESTING PURPOSES - can call the chart_pipeline(company_dict) function directly in the main pipeline
# Estimated runtime per report = ~30 secs
# file_path = 'DBS2020.json'
# f = open(file_path,)
# data = json.load(f)
# result = chart_pipeline(data)
