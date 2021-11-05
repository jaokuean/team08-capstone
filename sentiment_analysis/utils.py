import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np


def get_sentiment_score(sentence):
    sid = SentimentIntensityAnalyzer()

    update_lexicon_esg = {
        'initiate': 1,
        'initiatives':1,
        'energy-saving': 1,
        'solar': 1,
        'transition': 1,
        'low-carbon': 1,
        'reduction': 1,
        'reducing': 1,
        'achieve': 1,
        'achieved': 1,
        'achievement':1,
        'new': 1,
        'reduce':1,
        'waste':0,
        'energy':0,
        'wind':1,
        'decrease':1,
        'increase':0,
        'asset':0,
        'assets':0,
        'green':1,
        'recycle':1,
        'recycled':1,
        'recycling':1
    }

    sid.lexicon.update(update_lexicon_esg)
    return sid.polarity_scores(sentence)['compound']

def proc_temp_sentiments(temp_results):
    return [i[1]/i[0] if i[0] != 0 else None for i in temp_results]