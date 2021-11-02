import os
import re
import spacy
import en_core_web_sm
import string
SPECIAL_EXT_TOKENS = ['double', 'doubled', 'triple', 'tripled', 'half', 'quarter']

def get_indices_filter_nondigits(data, filter_corpus):
    # this processed json file, flattens and store indices for preprocessed text for easier retrieval of original text later
    d = {}
    for company in data:
        d[company['url']] = []
        for page_ind in range(len(company['report_sentences_preprocessed'])):
            for sentence_ind in range(len(company['report_sentences_preprocessed'][page_ind])):
                sentence = company['report_sentences_preprocessed'][page_ind][sentence_ind]
                if line_has_digits(sentence) and line_has_decarbonisation(sentence, filter_corpus):
                    text = preprocess(sentence, False)
                    d[company['url']].append((sentence, text, (page_ind, sentence_ind)))
    return d

def line_has_digits(sentence): 
    # this filters out those lines with possible metrics
    line = remove_year_co2(sentence)
    for j in line:
        if j.isdigit(): # this step filters 25% of the data
            return True

def line_has_decarbonisation(sentence, filter_corpus): #replace with word embeddings later on
    words = sentence.split(" ")
    for i in words:
        if i in filter_corpus:
            return True

# preproc for pos-tagging
def preprocess(text, lemmatization_bool):
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
    if lemmatization_bool:
        removed = " ".join(lemmatization(removed.split(" ")))
    removed = removed.strip()
    return removed 

# this removes years from the string (prevent inteference with pos extraction)
def remove_year_co2(text):
    removed = preprocess(text, False)
    removed = removed.replace("co2", "")
    return removed

# this extracts out the pos tag for each token in thhe cleaned words
def pos_extraction(text):
    nlp = spacy.load("en_core_web_sm")
    tokens, pos, tag = [], [], []
    doc = nlp(text)
    for token in doc:
        tokens.append(token)
        pos.append(token.pos_)
        tag.append(token.tag_)
    return tokens, pos, tag 

# function to tag both original and processed versions of text
def tag_both_text(org_text, processed_text):
    tk_org, pos_org, tag_org = pos_extraction(org_text)
    tk_proc, pos_proc, tag_proc = pos_extraction(processed_text)
    return [org_text, processed_text, tk_org, pos_proc, tag_proc]

# helper function to print output of pos tagging
def pos_printer(token, pos, tag):
    for i in range(len(token)):
        print(str(token[i])+' -> ' + pos[i] +',' +tag[i])

def lemmatization(text_list, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    nlp_lemma = spacy.load("en_core_web_sm", disable=['ner'])
    texts_out = []
    for texts in text_list:
        texts_out.append(" ".join([token.lemma_ for token in nlp_lemma(texts)]))
    return texts_out

def process_list(test_set_preproc):
    tags = []
    for i,j in test_set_preproc: 
        tk_org, pos_org, tag_org = pos_extraction(i)
        tk, pos, tag = pos_extraction(j)
        tags.append([i, j, tk_org, pos, tag]) 
    return tags

def extract_text(tags, verb_exclude): 
    tokens, pos_list, tag_list = tags[2], tags[3], tags[4]
    results = []
    for i in range(len(pos_list)):
        pos = pos_list[i]
        tag = tag_list[i]
        tok = tokens[i]
        if pos == 'NUM' and line_has_digits(tokens[i].text): #million recognised as a NUM
            j,k = extract_text_numbers(pos_list, tag_list, verb_exclude, i)
            results.append([tokens[i].text, generate_extracted_text(tokens, pos_list, j,k)])
        if (pos == 'DET' and tag == 'PDT') or tok.text in SPECIAL_EXT_TOKENS:
            j,k = extract_text_quant_words(pos_list, tag_list, i)
            results.append([tokens[i].text, generate_extracted_text(tokens, pos_list, j,k)])
    return results    

def extract_text_numbers(pos_list, tag_list, verb_exclude, i):
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
    
def generate_extracted_text(tokens, pos_list, j, k): #need to write if not simply joining will give extra spaces
    extracted_text = ''
    for tk in range(j,k):
        if pos_list[tk] != 'PUNCT' and pos_list[tk] != 'PART':
            extracted_text += ' '
        extracted_text += tokens[tk].text
    return extracted_text.strip()