from .utils import *

def sentiment_analysis_pipeline(df, text_class_pred):
    """
    Calculates the average sentiment within each text classification class using VADER.

    Parameters
    ----------
    df : dataframe
        Dataframe representation of predicted relevant text
    text_class_pred: list
        Gives the corresponding predicted text class of each sentence found in df

    Return
    ------
    final_results : list
        Gives average sentiment within each carbon category (the label matches the list index). 
        None if there are no records predicted for that particular class.

    """   
    temp_results = [[0,0],[0,0],[0,0],[0,0],[0,0]]
    all_sentences = list(df.cleaned_sentence)

    for i in range(len(all_sentences)):
        sentence = all_sentences[i]
        pred_class = text_class_pred[i]
        pred_sentiment = get_sentiment_score(sentence)
        index = 4
        if pred_class == 'Carbon Emissions':
            index=0
        elif pred_class == 'Energy':
            index=1
        elif pred_class == 'Waste':
            index=2
        elif pred_class == 'Sustainable Investing':
            index=3
        temp_results[index][0] += 1
        temp_results[index][1] += pred_sentiment
    final_results = proc_temp_sentiments(temp_results)
    return final_results