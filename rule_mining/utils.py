import os
import re
import spacy
import string
SPECIAL_EXT_TOKENS = ['double', 'doubled', 'triple', 'tripled', 'half', 'quarter']
PUNC = [x for x in string.punctuation]

def line_has_digits(sentence): 
    """
    Checks if a sentence has digits

    Parameters
    ----------
    Sentence: Str
        Sentence we are checking

    Return
    ------
    Bool
        Whether sentence has digits or not

    """  
    # this filters out those lines with possible metrics
    line = remove_year_co2(sentence)
    for j in line:
        if j.isdigit(): # this step filters 25% of the data
            return True

def preprocess(text, lemmatization_bool):
    """
    Preprocesses a sentence for pos-tagging. Removes most date formats so it does not interfere with rule mining
    
    Parameters
    ----------
    text: Str
        Sentence we are processing
    lemmatization_bool: Bool
        Indicates whether we want to apply lemmatization to the input sentence

    Return
    ------
    removed: Str
        Processed sentence

    """  
    removed = re.sub(r'(\d{2})/(\d{2})/(\d{4})', 'date_dummy', text) # replace with dummy
    removed = re.sub(r'(jan|february|march|april|may|june|july|august|september|october|november|december) ([1-3]{2}), ([1-2][0-9]{3})', 
                     'dummy dummy dummy', text)
    removed = re.sub(r'(jan|february|march|april|may|june|july|august|september|october|november|december) ([1-3]{2}) ([1-2][0-9]{3})', 
                     'dummy dummy dummy', text)
    removed = re.sub(r'([1-2][0-9]{3}) (jan|february|march|april|may|june|july|august|september|october|november|december) ([1-3]{2})', 
                     'dummy dummy dummy', text)
    removed = re.sub(r'([1-3]{2}) (jan|february|march|april|may|june|july|august|september|october|november|december), ([1-2][0-9]{3})', 
                     'dummy dummy dummy', text)
    removed = re.sub(r'([1-3]{2}) (jan|february|march|april|may|june|july|august|september|october|november|december) ([1-2][0-9]{3})', 
                     'dummy dummy dummy', text)
    removed = re.sub(r'(\d{4})/(\d{4})', 'year_dummy', removed)
    removed = re.sub(r'[1-2][0-9]{3}', 'year_dummy', removed)
    removed = re.sub(r'.?-.?', 'xx_', removed)
    if lemmatization_bool:
        removed = " ".join(lemmatization(removed.split(" ")))
    removed = removed.strip()
    return removed 

def remove_year_co2(text):
    """
    This removes years & CO2 from the string (prevent inteference with pos extraction)
    
    Parameters
    ----------
    text: Str
        Sentence we are processing

    Return
    ------
    removed: Str
        Processed sentence without co2 and year

    """
    removed = preprocess(text, False)
    removed = removed.replace("co2", "")
    return removed

def pos_extraction(text):
    """
    This extracts out the pos tag for each token in thhe cleaned words
    
    Parameters
    ----------
    text: Str
        Sentence we are processing

    Return
    ------
    tokens: List[Spacy Token]
        Spacy token of each word in the sentence
    pos: List[Str]
        Coarse grain part of speech of each corresponding token
    tag: List[Str]
        Fine grain part of speech of each corresponding token
    """
    nlp = spacy.load("en_core_web_sm")
    tokens, pos, tag = [], [], []
    doc = nlp(text)
    for token in doc:
        tokens.append(token)
        pos.append(token.pos_)
        tag.append(token.tag_)
    return tokens, pos, tag 

def lemmatization(text_list):
    """
    Lemmatizates the token inputs https://spacy.io/api/annotation
   
    Parameters
    ----------
    text_list: List[Spacy Token]
        Spacy token of each word in the sentence we want to lemmatize

    Return
    ------
    texts_out: List[Str]
        Lemmatized version of each token in list representation

    """ 
    nlp_lemma = spacy.load("en_core_web_sm", disable=['ner'])
    texts_out = []
    for texts in text_list:
        texts_out.append(" ".join([token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in nlp_lemma(texts)]))
    return texts_out

def process_list(test_set_preproc):
    """
    Performs part of speech tagging on the original and lemmatize sentence
   
    Parameters
    ----------
    test_set_preproc: List[Str]
        List consisting of 2 objects
        First object gives the original sentence without any preprocessing
        Second object gives the cleaned sentence after running through preprocess()

    Return
    ------
    [i, j, tk_org, pos, tag]: List[Str, Str, List[Spacy Token], List[Str], List[Str]
        First object is the original sentence
        Second object is the cleaned sentence after running through preprocess()
        Third object gives a list of the spacy tokens found in the original sentence
        Fourth object gives the coarse grain part of speech for each token in the cleaned sentence
        Last object gives the fine grain part of speech for each token in the cleaned sentence

    """ 
    i,j = test_set_preproc[0], test_set_preproc[1]
    tk_org, pos_org, tag_org = pos_extraction(i)
    tk, pos, tag = pos_extraction(j)
    return [i, j, tk_org, pos, tag]

def extract_text(tags, verb_exclude): 
    """
    Extracts the text using devised rule mining algorithms
    Using the extracted text, format the original sentence into a markdown friendly format for visualisation on dashboard
   
    Parameters
    ----------
    tags: List[Str, Str, List[Spacy Token], List[Str], List[Str]
        First object is the original sentence
        Second object is the cleaned sentence after running through preprocess()
        Third object gives a list of the spacy tokens found in the original sentence
        Fourth object gives the coarse grain part of speech for each token in the cleaned sentence
        Last object gives the fine grain part of speech for each token in the cleaned sentence

    verb_exclude: List[Str]
        Fine grain part of speech to exclude for number extraction

    Return
    ------
    results: Str
        Formatted text of the the original sentence into a markdown friendly format for visualisation on dashboard
    """ 
    tokens, pos_list, tag_list = tags[2], tags[3], tags[4]
    results = ''
    start,end = 0, min(len(tokens), len(pos_list))
    for i in range(min(len(tokens), len(pos_list))):
        pos = pos_list[i]
        tag = tag_list[i]
        tok = tokens[i]
        if pos == 'NUM' and line_has_digits(tokens[i].text): #million recognised as a NUM
            j,k = extract_text_numbers(pos_list, tag_list, verb_exclude, i)
            start = max(start, j)
            end = min(end, k)
            results = generate_extracted_text(tokens, pos_list, j,k)
            break
        if tok.text in SPECIAL_EXT_TOKENS:
            j,k = extract_text_quant_words(pos_list, i)
            start = max(start, j)
            end = min(end, k)
            results = generate_extracted_text(tokens, pos_list, j,k)
            break
    if (end-start) >= int(0.9* len(tokens)):
        results = format_extracted_text_toolong(tags[0])
    return results    

def extract_text_numbers(pos_list, tag_list, verb_exclude, i):
    """
    Extracts the text using devised rule mining algorithms for numbers
   
    Parameters
    ----------
    pos_list: List[Str]
        Gives the coarse grain part of speech for each token in the cleaned sentence
    tag_list: List[Str]
        Gives the fine grain part of speech for each token in the cleaned sentence
    verb_exclude: List[Str]
        Fine grain part of speech to exclude for number extraction
    i: Int
        Gives the index of the token we are extracting

    Return
    ------
    j: Int
        Starting index of word we are extracting from
    k: Int
        Ending index of word we are extracting till
    """
    j = max(i-1,0)
    k = min(i+1, len(pos_list)-1)    
    noun_flag_left, noun_flag_right = False, False
    if j != 0:
        while pos_list[j] != 'VERB' or noun_flag_left == False or tag_list[j] in verb_exclude:
            if pos_list[j] == 'NOUN':
                noun_flag_left = True
            j -= 1
            if j == 0:
                break
    if k != len(pos_list)-1:
        while pos_list[k] != 'VERB' or noun_flag_right == False or tag_list[k] in verb_exclude:
            if pos_list[k] == 'NOUN':
                noun_flag_right = True
            k += 1
            if k == len(pos_list)-1:
                break
    return j,k

def extract_text_quant_words(pos_list, i):
    """
    Extracts the text using devised rule mining algorithms for quantifier words
   
    Parameters
    ----------
    pos_list: List[Str]
        Gives the coarse grain part of speech for each token in the cleaned sentence
    i: Int
        Gives the index of the token we are extracting

    Return
    ------
    j: Int
        Starting index of word we are extracting from
    k: Int
        Ending index of word we are extracting till
    """
    j = max(i-1,0)
    k = min(i+1, len(pos_list)-1)    
    adj_flag_left, adj_flag_right = False, False
    if j != 0:
        while pos_list[j] != 'NOUN' or adj_flag_left == False:
            if pos_list[j] == 'ADJ':
                adj_flag_left = True
            j -= 1
            if j == 0:
                break
    if k != len(pos_list)-1:
        while pos_list[k] != 'NOUN' or adj_flag_right== False:
            if pos_list[k] == 'ADJ':
                adj_flag_right = True
            k += 1
            if k == len(pos_list)-1:
                break
    return j,k

def generate_extracted_text(tokens, pos_list, j, k):
    """
    Generates the final extracted text in string form. 
    Need this if not simply joining will give extra spaces/not formatted properly  
   
    Parameters
    ----------
    tokens: List[Spacy Tokens]
        Spacy token of each word in the original sentence
    pos_list: List[Str]
        Gives the coarse grain part of speech for each token in the cleaned sentence
    j: Int
        Starting index of word we are extracting from
    k: Int
        Ending index of word we are extracting till

    Return
    ------
    extracted_text: Str
        Original text formatted with markdown asterisks to bold the portions that were extracted from the rule mining algorithm
    """
    extracted_text = ''
    for tk_ind in range(len(tokens)):
        if tokens[tk_ind].text not in PUNC:
            extracted_text += ' '
        if tk_ind == j:
            extracted_text += ' **'
        extracted_text += tokens[tk_ind].text  
        if tk_ind == k:
            extracted_text += '** '
    return extracted_text.strip()

# case where we need to bold numbers only
def format_extracted_text_toolong(org_sentence): 
    """
    This formats the original sentence with markdown asterisks to bold numbers
    This is done should the extracted text be >90% of the original sentence, which makes bolding >90% of the sentence make no sense
   
    Parameters
    ----------
    org_sentence: Str
        Original Sentence

    Return
    ------
    extracted_text: Str
        Original text formatted with markdown asterisks to bold all numbers in the sentence
    """
    org_tk, org_pos, org_tag = pos_extraction(org_sentence)
    extracted_text = ''
    pattern = re.compile("\d{4}")
    for ind in range(len(org_tk)):
        tk = org_tk[ind]
        if tk.text not in PUNC:
            extracted_text += ' '
        if pattern.match(tk.text) or tk.pos_ != 'NUM': 
            extracted_text += tk.text
        else:
            extracted_text += '**' + tk.text + '** '
    return extracted_text.strip()