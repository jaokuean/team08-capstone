from .utils import *

def sentiment_analysis_pipeline(df, text_class_pred):
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