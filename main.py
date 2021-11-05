import json
import pandas as pd
import ast
from text_classification.text_classification_pipeline import *
from rule_mining.rule_mining_pipeline import *
from sentiment_analysis.sentiment_analysis_pipeline import *
from word_cloud.word_cloud_pipeline import *

def text_pipeline(json_path):
    # Opening JSON file
    f = open(json_path,)

    # returns JSON object as a dictionary
    data = json.load(f)

    # Loop through for each company
    print("Running through Companies")
    for i in data:
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

        # Relevance Prediction - XM maybe you wanna put ur text portion here

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

    # If don't need this, we can just return the dictionary data; to link to image & tables & charts portion
    # print('Writing Data')
    # with open('data/test_fin.json', 'w') as fp:
    #     json.dump(data,fp)

if __name__=='__main__':
    text_pipeline('data/all_text_chart_output.json')