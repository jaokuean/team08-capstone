import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np


def get_sentiment_score(sentence):
    """
    Calculates the polarity sentiment for a sentence using VADER.

    Parameters
    ----------
    sentence : str
        Sentence we would like to retrieve polarity sentiment for

    Return
    ------
    polarity score : float
        Polarity sentiment score for the sentence

    """   
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
    """
    Processes and calculates the average sentiment within each category

    Parameters
    ----------
    temp_results : list[list]
        Contains the temporary results in the previous for loop. 
        It is a list of lists, with each inner list giving [number of predicted sentences in that class, total polarity sentiment]

    Return
    ------
    results : list
        Polarity sentiment score for each decarbonisation category. Index corresponds to the class; i.e. 0 corresponds to decarbonisation

    """   
    return [i[1]/i[0] if i[0] != 0 else None for i in temp_results]