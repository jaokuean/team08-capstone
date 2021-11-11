import json
import pandas as pd
import ast
import pickle5 as pickle
import shutil
import os

# data collection, page and bert sentence filtering
from text_filtering.data_collection_filtering_pipeline import *
# # # relevance prediction
from relevance_prediction.relevance_prediction_pipeline import *
# # # carbon class prediction
from text_classification.text_classification_pipeline import *
from rule_mining.rule_mining_pipeline import *
from sentiment_analysis.sentiment_analysis_pipeline import *
from word_cloud.word_cloud_pipeline import *

# table detection
from table_extraction.table_pipeline import *
# chart detection 
from chart_extraction.chart_extraction import *


################################### Helper Functions ################################### 
# Text Classification
def text_except_relevance(json_path):
    """
    Function that runs carbon class prediction pipeline, sentiment analysis pipeline, word cloud generation pipeline and sentiment analysis pipeline for 1 FI after running it through the relevance prediction pipeline.

    Parameters
    ----------
    json_path : str
        String of path to a json file containing company's report details, "text_output" field which includes pages of relevance sentences, relevant sentences predicted by the model and probability scores. 

    Return
    ------
    output_path : str
        String of path to output file containing "text_output" field with carbon class, sentiment analysis, word cloud and mined text.
    """  
    
    # Opening JSON file    
    f = open(json_path,)

    # returns JSON object as a dictionary
    data = json.load(f)

    # Loop through for each company
    print("Text Classification Pipeline")
    i = data
    try:
        text_data = i['text_output']['sentence']
    except TypeError:
        temp_dict = ast.literal_eval(i['text_output'])
        new_dict = {}
        for k,v in temp_dict.items():
            new_dict[k] = list(v.values())
        i['text_output'] = new_dict
        text_data = new_dict['sentence']
    finally:
        df = pd.DataFrame(text_data, columns = ['sentence'])
        df['cleaned_sentence'] = df['sentence'].apply(clean_sentence)

    # Text Classification
    print("Generating Text Classification Predictions")
    text_class_pred = text_classification_pipeline(df)

    # Save Text Classification Predictions
    i['text_output']['carbon_class'] = text_class_pred

    # # Rule Mining
    print("Generating Rule Mining Text")
    mined_text = rule_mining_pipeline(df)

    # # Save Mined Text
    i['text_output']['mined_text'] = mined_text

    # Word Cloud
    print("Generating Word Clouds")
    wordcloud_img_path = word_cloud_pipeline(df.sentence, text_class_pred, i['company'], i['year'])

    # Save Word Cloud Image Paths
    i['wordcloud_img_path'] = wordcloud_img_path

    # Sentiment Analysis
    print("Generating Sentiments")
    class_sentiments = sentiment_analysis_pipeline(df ,text_class_pred)
    i['sentiment_score'] = class_sentiments

    print('Writing Data')
    output_path = json_path[:-5] + "_all.json"
    
    with open(output_path, 'w') as fp:
         json.dump(i,fp)
    
    return output_path


# Appending new report to database 
def combine_intermediate_json(all_json_paths): # input [text,table,chart]
    """
    Function that combines all intermediate output files from text extraction, table extraction and chart extraction, into 1 json.

    Parameters
    ----------
    json_path : list of str
        List containing paths to the various intermediate output files.
    Return
    ------
    output_path : str
        String of path to output file containing combined data from all 3 files.
    """ 
    
    all_json = []
    # open files
    for paths in all_json_paths:
        with open(paths,'r') as infile:  
            all_json.append(json.load(infile))
    
    
    # append text data
    final_json = all_json[0]
    
    # append table data
    final_json["table_keywords"] = all_json[1]["table_keywords"]
    final_json["table_image_keywords"] = all_json[1]["table_image_keywords"]
    table_images_paths = all_json[1]["table_images"]
    if table_images_paths != "nan":
        table_images_paths_final = {}
        for page, table_image_path_list in table_images_paths.items():
            path_list = []
            for path in table_image_path_list:
                path_final = re.sub("new_report","dashboard_data",path)
                path_list.append(path_final)
            table_images_paths_final[page] = path_list      
        final_json["table_images"] = table_images_paths_final
    else:
        final_json["table_images"] = all_json[1]["table_images"]


    # append  chart data
    final_json["chart_images_keywords"] = all_json[2]["chart_images_keywords"]
    chart_images_paths = all_json[2]["chart_images"]
    chart_images_paths_final = {}
    if chart_images_paths != "nan":
        for page, chart_image_path_list in chart_images_paths.items():
            path_list = []
            for path in chart_image_path_list:
                path_final = re.sub("new_report","dashboard_data",path)
                path_list.append(path_final)
            chart_images_paths_final[page] = path_list      
        final_json["chart_images"] = chart_images_paths_final
    else:
        final_json["chart_images"] = all_json[2]["chart_images"]
    
    output_path = all_json_paths[2][:-12] + "_FINAL.json"
                                                      
    with open(output_path, 'w') as output_file:
        json.dump(final_json, output_file)   
    
    return output_path




def append_json_to_database(file_path):
    
    """
    Function that appends the json containing combined data from all intermediate files to the database json file.

    Parameters
    ----------
    file_path : list of str
        String of paths to the json file containing combined data
    Return
    ------
    None
    """
       
    with open(file_path,'r') as infile:
        new_entry = json.load(infile)
    
    database_path = "assets/data/dashboard_data/final_database.json"  
    
    with open(database_path,'r') as infile:  
        database = json.load(infile)
    
    database.append(new_entry)
    
    with open(database_path,'w') as output_file:  
        json.dump(database, output_file)
        


def append_pickle_to_database(file_path):
    
    """
    Function that appends the pickle file of the new report to the database pickle file.

    Parameters
    ----------
    file_path : str
        String of path to the pickle file of the new report
    Return
    ------
    None
    """  
    
    with open(file_path, 'rb') as input_pickle:
        new_pickle = pickle.load(input_pickle)

    database_path = "assets/data/dashboard_data/tbl_ALL.pickle"  

    with open(database_path, 'rb') as input_pickle:
        database = pickle.load(input_pickle)

    database.append(new_pickle)

    with open(database_path,"wb") as outpickle:
        pickle.dump(database,outpickle,protocol=pickle.HIGHEST_PROTOCOL)

    


        
def append_images_to_database():
    """
    Function that moves output images from new report (source) to the database folders (target). images at the source files will be deleted.

    Parameters
    ----------
    None
    
    Return
    ------
    None
    """ 
    
    # moves file from source to target_dir and deletes files from source
    folders = ["wordcloud_images","ChartExtraction_Output","table_images"]
    new_report_folder = "data/new_report/"
    database_folder = "assets/data/dashboard_data/" 
    
    for folder in folders:
        source_dir = new_report_folder + folder
        target_dir = database_folder + folder

        file_names = os.listdir(source_dir)
        print(file_names)

        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name), os.path.join(target_dir,file_name))

def delete_intermediate_files():
    """
    Function that deletes all intermediate and final json in the new_report directory. Clears the directory after data for the new report is added to database.

    Parameters
    ----------
    None
    
    Return
    ------
    None
    """  
    directory = "data/new_report/"
    filelist = [ f for f in os.listdir(directory) if f.endswith(".json") or f.endswith(".pkl")]
    for f in filelist:
        os.remove(os.path.join(directory, f))    

        
        


################################### Main Function ################################### 
def new_url_run(report_url,report_company,report_year,downloaded=False):   
    """
    Main Function to run to whole information extraction pipeline for a new report. IntermediatenNew report files are created in data/new_report/ and subsquently moved to assets/data/dashboard_data/ and appended to assets/data/dashboard_data/final_database.json and tbl_ALL.pickle.
    
    Parameters
    ----------
    report_url: str
        If downloaded=False, report is from the internet and report_url is a URL string to the PDF.
        If downloaded=True, report is from local machine and report_url is the file path to the PDF.
    
    report_company: str
        Company name
    
    report_year: str
        Year of report
        
    downloaded : bool
        Whether report needs to be downloaded from Internet or not
        
    Return
    ------
    None
    
    """
    
    # data collection
    ## new json generated in "data/new_report" 
    report_output_file_path = upload_pdf(report_url,report_company,report_year,downloaded)
    # check if PDF could be collected, throw exception if it cannot
    if report_output_file_path == "":
        raise AttributeError
           
    
    # text extraction
    ## new BERT_embeddings_json generated in "data/new_report" 
    report_bert_output_file_path = bert_filtering(report_output_file_path)
    #report_bert_output_file_path = 'data/new_report/Canada Pension2017_BERT_embeddings.json'
    
    ## relevance prediction 
    text_output_path = relevance_prediction(report_bert_output_file_path)
    # check if got relevant sentences, throw exception if there are none
    if text_output_path == "":
        #delete intermediate files
        delete_intermediate_files()
        raise ValueError
        
    ## all other text predictions 
    #text_output_path = "data/new_report/Canada Pension2017_text_output.json"
    all_text_output_path = text_except_relevance(text_output_path)
    
   
      # table extraction
#     report_output_file_path = "data/new_report/Canada Pension2017.json"

    table_output_path, table_output_pickle_path = table_pipeline(report_output_file_path)
    
    # chart detection 
#     report_output_file_path = "data/new_report/Canada Pension2017.json"
    chart_output_path = chart_pipeline(report_output_file_path)
    
    # combine all data into database
    print("APPENDING NEW REPORT TO DATABASE")

    all_json = [all_text_output_path,table_output_path,chart_output_path]
    final_output_path = combine_intermediate_json(all_json)
    append_json_to_database(final_output_path)
    append_pickle_to_database(table_output_pickle_path)
    append_images_to_database() #word cloud, charts, tables
    
    # clear the new_report folder for new report next time
    delete_intermediate_files() # delete all json files, keeping the empty wordcloud, chart, table folders
    
    
        
    
    
    



# test functiona call

if __name__ == "__main__":
    report_urls = ["https://www.cppinvestments.com/wp-content/uploads/2019/10/CPPIB_SI_Report_ENG.pdf",
                   "https://www.gam.com/-/media/content/corporate-responsibility/gam-responsible-investment-policy.pdf"]
    report_companys = ["Canada Pension", "GAM"]
    report_years = ["2017", "2020"]

    for i in range(len(report_urls)):
        print(report_companys[i])
        new_url_run(report_urls[i], report_companys[i],
                    report_years[i], downloaded=False)
        print(f"DONE WITH {report_companys[i]}")