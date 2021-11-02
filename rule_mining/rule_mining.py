from .utils import preprocess, process_list, extract_text

def process_json_rulemining(json_file):       
    # Iterating through the json
    for i in json_file:
        tagged = []
        text_data = i['text_output']['sentence']

        # Loop through all relevant sentences and append [Original Sentence, Lemmatized Sentence] to tagged
        for j in text_data:
            tagged.append([j.lower(), preprocess(j.lower(), True)])
        
        # Process the list of [Original Sentence, Lemmatized Sentence] to 
        # [Original Sentence, Lemmatized Sentence, Original Token, Processed Coarse Tag, Processed Fine Tag]
        processed_tags = process_list(tagged)

        extracted_texts = [] 
        # Call extraction algorithm to retrieve [[token, text], .....]
        for j in processed_tags:
            extracted_texts.append(extract_text(j, ['VBP']))
        
        # Append back to JSON
        i['text_output']['mined_text'] = extracted_texts
    return json_file
