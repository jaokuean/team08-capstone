import re, numpy as np, pandas as pd
from pprint import pprint

# Gensim 
import gensim, spacy, logging, warnings
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'bank', 'project', 'company', 'employee', 'head', 'report', 'subject', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come', 'report', 'page'])

# wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

#%matplotlib inline
warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


def sent_to_words(sentences):
    """
    Tokenize sentences to words and lowercase.

    Parameters
    ----------
    sentences : list of str
        List of sentences.

    Yield
    ------
    sent : list of str
        List of tokens.
    """ 
    for sent in sentences:
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)

def process_words(texts, bigram_mod, trigram_mod, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB']):
    """
    Remove stopwords, form Bigrams, Trigrams and lemmatization
    
    Parameters
    ----------
    texts : list of str
        List of sentences.
    bigram_mod : obj
        Bigram model constructed.
    trigram_mod : obj
        Trigram model constructed.
    stop_words : list of str
        List of stop words.
    allowed_postags : list of str
        List of allowed postags.

    Return
    ------
    texts_out : list of list of str
        List of list of preprocessed words.
    """
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out

def prepare_data(data):
    """
    Process text data.

    Parameters
    ----------
    data : list of str
        List of sentences.

    Return
    ------
    data_processed : list of list of str
        List of list of processed words.
    """ 
    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=2, threshold=50) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    data_processed = process_words(data_words, bigram_mod, trigram_mod)  # processed Text Data
        
    return data_processed

def generate_wordcloud(data, path, carbon_class):
    """
    Generate wordcloud based on process texts.

    Parameters
    ----------
    data : list of list of str
        List of list of processed words.
    path : str
        Image path of wordcloud to be saved in.
    carbon_class : str
        One of the 4 carbon classes.
    """ 
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    
    if carbon_class == 'Carbon Emissions':
        i = 0
    elif carbon_class == 'Energy':
        i = 1
    elif carbon_class == 'Waste':
        i = 2
    else:
        i = 3

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=1800,
                      height=1800,
                      max_words=10,
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)
    
    flat_list = [item for sublist in data for item in sublist]
    word_frequency_list = pd.Series(flat_list).value_counts()
    final = dict(word_frequency_list)
    cloud.generate_from_frequencies(final, max_font_size=300)
    plt.figure(figsize=(6,6))
    plt.imshow(cloud)
    plt.title(carbon_class, fontdict=dict(size=24))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return
