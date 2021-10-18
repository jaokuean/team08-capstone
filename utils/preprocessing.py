import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
import string 

STOPWORD_SET = list(STOPWORDS.union(set(stopwords.words("english"))))
WORD_TOKENIZER = nltk.WordPunctTokenizer()
LEMMATIZER = WordNetLemmatizer()
PUNCTUATION_TABLE = str.maketrans(dict.fromkeys(string.punctuation))


def clean_sentence(sentence,remove_stopwords=True, remove_punctuation=True, remove_numbers=True, lemmatize=True):

    if remove_stopwords:
        sentence = " ".join([word for word in WORD_TOKENIZER.tokenize(sentence) if not word in STOPWORD_SET])
    if remove_punctuation:
        sentence = sentence.translate(PUNCTUATION_TABLE)
    if remove_numbers:
        sentence = "".join([i for i in sentence if not i.isdigit()])
    if lemmatize:
        sentence = " ".join([LEMMATIZER.lemmatize(word) for word in WORD_TOKENIZER.tokenize(sentence)])
    
    sentence = " ".join(sentence.split())
    return sentence.lower()

