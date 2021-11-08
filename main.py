import json
import pandas as pd
import ast

# data collection, page and bert sentence filtering
from text_filtering.data_collection_filtering_pipeline import *
# relevance prediction
from relevance_prediction.relevance_prediction_pipeline import *
# carbon class prediction
from text_classification.text_classification_pipeline import *
from rule_mining.rule_mining_pipeline import *
from sentiment_analysis.sentiment_analysis_pipeline import *
from word_cloud.word_cloud_pipeline import *

################################### helper function ################################### 
# for all carbon classes predictions
def text_except_relevance(json_path):
    # Opening JSON file
    
    f = open(json_path,)

    # returns JSON object as a dictionary
    data = json.load(f)

    # Loop through for each company
    print("Text Classification Pipeline")
    i = data
    print(i['company'])
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




################################### main function ################################### 
def new_url_run(report_url,report_company,report_year,downloaded=False):
    # data collection
    ## new json generated in "data/sustainability_reports_new" -OK
    #report_output_file_path = upload_pdf(report_url,report_company,report_year,downloaded)
        
    # text
    ## new BERT_embeddings_json generated in "data/sustainability_reports_new"
    #report_bert_output_file_path = bert_filtering(report_output_file_path)
    report_bert_output_file_path = 'data/sustainability_reports/new/Canada Pension2017_BERT_embeddings.json'
    
    ## relevance prediction - OK
    text_output_path = relevance_prediction(report_bert_output_file_path)
    ## all other text predictions
    all_text_output_path = text_except_relevance(text_output_path)
    
   
    # tables
    




# test functiona call

if __name__ == "__main__":
    report_url = "https://www.cppinvestments.com/wp-content/uploads/2019/10/CPPIB_SI_Report_ENG.pdf"
    report_company = "Canada Pension"
    report_year = "2017"
    new_url_run(report_url,report_company,report_year,downloaded=False)

    







