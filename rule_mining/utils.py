import os
import re
import spacy
import string
SPECIAL_EXT_TOKENS = ['double', 'doubled', 'triple', 'tripled', 'half', 'quarter']
PUNC = [x for x in string.punctuation]

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
    removed = re.sub(r'.?-.?', 'xx_', removed)
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

def lemmatization(text_list):
    """https://spacy.io/api/annotation"""
    nlp_lemma = spacy.load("en_core_web_sm", disable=['ner'])
    texts_out = []
    for texts in text_list:
        texts_out.append(" ".join([token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in nlp_lemma(texts)]))
    return texts_out

def process_list(test_set_preproc):
    i,j = test_set_preproc[0], test_set_preproc[1]
    tk_org, pos_org, tag_org = pos_extraction(i)
    tk, pos, tag = pos_extraction(j)
    return [i, j, tk_org, pos, tag]

def extract_text(tags, verb_exclude): 
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

# need to write if not simply joining will give extra spaces/not formatted properly  
def generate_extracted_text(tokens, pos_list, j, k):
    extracted_text = ''
    for tk_ind in range(len(tokens)):
        if tokens[tk_ind].text not in PUNC:
            extracted_text += ' '
        if tk_ind == j:
            extracted_text += '**'
        if tk_ind == k:
            extracted_text += '**'
        extracted_text += tokens[tk_ind].text  
    return extracted_text.strip()

# case where we need to bold numbers only
def format_extracted_text_toolong(org_sentence): 
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
            extracted_text += '**' + tk.text + '**'
    return extracted_text.strip()


