import pandas as pd
from .utils import *
import os

def word_cloud_pipeline(sentence, pred_class, company, year):
    # create a list to store 4 wordcloud image paths for 4 classes
    wordcloud_img_path = []

     # 4 fixed classes
    classes = ['Carbon Emissions', 'Energy', 'Waste', 'Sustainable Investing']

    # generating 1 word cloud for each class
    for c_class in classes:
        try:
            d = {'sentence': sentence, 'carbon_class': pred_class}
            df = pd.DataFrame(d)
            filtered_df = df[df['carbon_class'] == c_class]
            lst = filtered_df['sentence'].to_list()
            
            data_processed = prepare_data(lst)

            data_ready = []
            # remove words that are company names as they frequently occur but not meaningful
            for lst in data_processed:
                lst_of_words = []
                for word in lst:
                    if word not in company.lower():
                        lst_of_words.append(word)
                data_ready.append(lst_of_words)

            path = 'data/dashboard_data/wordcloud_images/' + company + '_' + year + '_' + c_class + '.png'
            output_path = 'data/new_report/wordcloud_images/' + company + '_' + year + '_' + c_class + '.png'
            # generate and save wordcloud image
            generate_wordcloud(data_ready, output_path, c_class)
            wordcloud_img_path.append(path)
            
        except Exception as e: # when there is no record in the carbon class
            print(e)
            path = 'data/dashboard_data/wordcloud_images/NO_DATA_' + c_class + '.png'
            wordcloud_img_path.append(path)

    return wordcloud_img_path
