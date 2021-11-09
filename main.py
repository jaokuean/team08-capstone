import json
import pandas as pd
import ast
import pickle
import shutil
import os

# data collection, page and bert sentence filtering
# from text_filtering.data_collection_filtering_pipeline import *
# # # relevance prediction
# from relevance_prediction.relevance_prediction_pipeline import *
# # # carbon class prediction
# from text_classification.text_classification_pipeline import *
# from rule_mining.rule_mining_pipeline import *
# from sentiment_analysis.sentiment_analysis_pipeline import *
# from word_cloud.word_cloud_pipeline import *

# table detection - AIFEN & JERMAINE COMMENT THIS OUT
#from table_extraction.table_pipeline import *
# chart detection - JK COMMENT THIS OUT
#from chart_extraction.chart_extraction import *


################################### Helper Functions ###################################
# Text Classification
def text_except_relevance(json_path):
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
        for k, v in temp_dict.items():
            new_dict[k] = list(v.values())
        i['text_output'] = new_dict
        text_data = new_dict['sentence']
    finally:
        df = pd.DataFrame(text_data, columns=['sentence'])
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
    wordcloud_img_path = word_cloud_pipeline(
        df.sentence, text_class_pred, i['company'], i['year'])

    # Save Word Cloud Image Paths
    i['wordcloud_img_path'] = wordcloud_img_path

    # Sentiment Analysis
    print("Generating Sentiments")
    class_sentiments = sentiment_analysis_pipeline(df, text_class_pred)
    i['sentiment_score'] = class_sentiments

    print('Writing Data')
    output_path = json_path[:-5] + "_all.json"

    with open(output_path, 'w') as fp:
        json.dump(i, fp)

    return output_path


# Database
def combine_intermediate_json(all_json_paths):  # input [text,table,chart]
    all_json = []
    # open files
    for paths in all_json_paths:
        with open(paths, 'r') as infile:
            all_json.append(json.load(infile))

    # append text data
    final_json = all_json[0]

    # append table data
#     final_json["table_keywords"] = all_json[1]["table_keywords"]
#     final_json["table_image_keywords"] = all_json[1]["table_image_keywords"]
#     final_json["table_images"] = all_json[1]["table_images"]

    # append  chart data
    final_json["chart_images_keywords"] = all_json[2]["chart_images_keywords"]
    final_json["chart_images"] = all_json[2]["chart_images"]

    output_path = all_json_paths[2][:-12] + "_FINAL.json"

    with open(output_path, 'w') as output_file:
        json.dump(final_json, output_file)

    return output_path


def append_json_to_database(file_path):
    with open(file_path, 'r') as infile:
        new_entry = json.load(infile)

    database_path = "data/dashboard_data_interim/final_database.json"  # CHANGE

    with open(database_path, 'r') as infile:
        database = json.load(infile)

    database.append(new_entry)

    with open(database_path, 'w') as output_file:
        json.dump(database, output_file)


def append_pickle_to_database(file_path):
    with open(file_path, 'rb') as input_pickle:
        new_pickle = pickle.load(input_pickle)

    database_path = "data/dashboard_data_interim/tbl_ALL.pickle"  # CHANGE

    with open(database_path, 'rb') as input_pickle:
        database = pickle.load(input_pickle)

    database.append(new_pickle)

    with open(database_path, "wb") as outpickle:
        pickle.dump(database, outpickle, protocol=pickle.HIGHEST_PROTOCOL)


def append_images_to_database():
    # moves file from source to target_dir and deletes files from source
    folders = ["wordcloud_images", "ChartExtraction_Output", "table_images"]
    new_report_folder = "data/new_report/"
    database_folder = "data/dashboard_data_interim/"  # CHANGE

    for folder in folders:
        source_dir = new_report_folder + folder
        target_dir = database_folder + folder

        file_names = os.listdir(source_dir)
        print(file_names)

        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name),
                        os.path.join(target_dir, file_name))


def delete_intermediate_files():
    directory = "data/new_report/"
    filelist = [f for f in os.listdir(directory) if f.endswith(".json")]
    for f in filelist:
        os.remove(os.path.join(mydir, f))


################################### Main Function ###################################
def new_url_run(report_url, report_company, report_year, downloaded=False):

    #     # data collection
    #     ## new json generated in "data/new_report" -OK IF REPORT CONTENT IS EMPTY, none is returned
    #     report_output_file_path = upload_pdf(report_url,report_company,report_year,downloaded)

    #     # text extraction
    #     ## new BERT_embeddings_json generated in "data/new_report" - OK
    #     #report_bert_output_file_path = bert_filtering(report_output_file_path)
    #     report_bert_output_file_path = 'data/new_report/Canada Pension2017_BERT_embeddings.json'

    #     ## relevance prediction - OK
    #     text_output_path = relevance_prediction(report_bert_output_file_path)
    #     ## all other text predictions - OK
    #     #text_output_path = "data/new_report/Canada Pension2017_text_output.json"
    #     all_text_output_path = text_except_relevance(text_output_path)

    #     table extraction -  AIFEN & JERMAINE COMMENT THIS OUT
    #     report_output_file_path = "data/new_report/Canada Pension2017.json"
    #     table_output_path, table_output_pickle_path = table_pipeline(report_output_file_path)

    #     # chart detection - JK COMMENT THIS OUT
    #     report_output_file_path = "data/new_report/Canada Pension2017.json"
    #     chart_output_path = chart_pipeline(report_output_file_path)

    #     # combine all data into database
    #     all_json = [all_text_output_path,table_output_path,chart_output_path]
    #     final_output_path = combine_intermediate_json(all_json)
    #     append_json_to_database(final_output_path)
    #     append_pickle_to_database(table_output_pickle_path)
    #     append_images_to_database() #word cloud, charts, tables

    #     # clear the new_report folder for new report next time
    #     delete_intermediate_files # delete all json files, keeping the empty wordcloud, chart, table folders

    # test functiona call
if __name__ == "__main__":
    report_url = "https://www.cppinvestments.com/wp-content/uploads/2019/10/CPPIB_SI_Report_ENG.pdf"
    report_company = "Canada Pension"
    report_year = "2017"
    new_url_run(report_url, report_company, report_year, downloaded=False)
