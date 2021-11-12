if __name__ == '__main__':
    from utils import preprocess, process_list, extract_text
else:
    from .utils import preprocess, process_list, extract_text
import pandas as pd
import re

def rule_mining_pipeline(df):    
    """
    Entry code to generate mined text from each sentence

    Parameters
    ----------
    df : dataframe
        Dataframe representation of predicted relevant text

    Return
    ------
    formatted_texts : list
        Gives markdown formatted text of original sentence. Mined portions will be enclosed in asterisks for bolding on dashboard

    """  
    formatted_texts = []
    text_data = list(df.sentence)

    # Loop through all relevant sentences 
    for j in text_data:

        # Create original sentence, lemmatized sentence
        org_lemma = [j, preprocess(j.lower(), True)]

        #Process the [Original Sentence, Lemmatized Sentence] to 
        # [Original Sentence, Lemmatized Sentence, Original Token, Processed Coarse Tag, Processed Fine Tag]
        processed = process_list(org_lemma)

        # Call extraction algorithm to retrieve formatted sentence for dashboard visualisation
        formatted_texts.append(extract_text(processed, ['VBP']))
        
    return formatted_texts

if __name__ == '__main__':
    b = ["A strategy that minimizes Scope 1 and 2 emissions will reduce exposure to power utilities who burn fossil fuels to generate electricity (Scope 1), but maintain or increase exposure to oil and gas producers, a sector where 80-90% of emissions are in Scope 3.", 
    "To ensure continuous reduction in water consumption, rules have been implemented for procurement, building utility replacements and new developments, thus allowing for long- term refinement.", "We have committed to using 100% renewable electricity by mid-2020.",
    "We have committed to using 100% renewable electricity by 2020.",
    "This will reduce our firms GHG footprint by 75% compared with 2004 levels.",
    "Consistent with BlackRocks goal to double offerings of sustainable ETFs (to 150), iShares launched over 45 new sustainable ETFs across the US, Europe, and Canada in 2020.",
    ]
    df = pd.DataFrame(b, columns = ['sentence'])
    print(rule_mining_pipeline(df))