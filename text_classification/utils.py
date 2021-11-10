import pandas as pd
import numpy as np

def get_sum_probs(df, heu_preds):
    """
    This sums all the model predicted probabilities of the sentence belonging to each class

    Parameters
    ----------
    df : DataFrame
        Has 25 columns; 5 for each model
        Dataframe containing probabilities of each model's prediction of a sentence being in that class.
    heu_preds: list
        Gives the heuristics' prediction of the sentence's class

    Return
    ------
    probs : DataFrame
        Each row corresponds to each sentence. 
        There are 5 columns; each giving the total model predicted probabilities of the sentence belonging to that class

    """   
    df['0_total'] = df['lr_prob_0'] + df['cb_prob_0'] + df['svm_prob_0'] + df['nb_prob_0'] + df['rf_prob_0']
    df['1_total'] = df['lr_prob_1'] + df['cb_prob_1'] + df['svm_prob_1'] + df['nb_prob_1'] + df['rf_prob_1']
    df['2_total'] = df['lr_prob_2'] + df['cb_prob_2'] + df['svm_prob_2'] + df['nb_prob_2'] + df['rf_prob_2']
    df['3_total'] = df['lr_prob_3'] + df['cb_prob_3'] + df['svm_prob_3'] + df['nb_prob_3'] + df['rf_prob_3']
    df['4_total'] = df['lr_prob_4'] + df['cb_prob_4'] + df['svm_prob_4'] + df['nb_prob_4'] + df['rf_prob_4']
    probs = df[['0_total','1_total','2_total','3_total','4_total']]
    for i in range(len(heu_preds)):
        pred = heu_preds[i]
        to_increase = '{pred}_total'.format(pred=pred)
        probs.at[i,to_increase] += 0.35
    return probs

def get_majority_pred_soft(df):
    """
    This gives the final predicted text class by finding the highest summed probabilities amongst the 5 for each sentence

    Parameters
    ----------
    df : DataFrame
        Each row corresponds to each sentence. 
        There are 5 columns; each giving the total model predicted probabilities of the sentence belonging to that class

    Return
    ------
    final_pred : List[Str]
        Each item consists of the final predicted class for each sentence

    """   
    final_pred = []
    for i in df.iterrows():
        lst = [j for j in i[1]]   
        max_value = max(lst)
        soft_pred = lst.index(max_value)
        if soft_pred == 0:
            final_pred.append('Carbon Emissions')
        elif soft_pred == 1:
            final_pred.append('Energy')
        elif soft_pred == 2:
            final_pred.append('Waste')
        elif soft_pred == 3:
            final_pred.append('Sustainable Investing')
        else:
            final_pred.append('Others')
    return final_pred

class_zero = ["emissions","footprint","ghg", "coal"]
class_one = ["energy","renewable","electricity","power", "solar", "kwh"]
class_two = ["waste","paper", "office","recycled","environmental"]
class_three = ["sustainable","investment","investments","bonds", "portfolio", "finance"]

def carbon_class_filter(row):
    """
    This is an implementation of the word heuristics classifier

    Parameters
    ----------
    row : Dictionary
        Each row corresponds to a predicted relevant sentence

    Return
    ------
    pred : int
        Returns the predicted text class of the sentence according to the heuristics classifier

    """ 
    sentence = row["sentence"]
    if any(map(sentence.__contains__, class_zero)):
        return 0
    elif any(map(sentence.__contains__, class_one)):
        return 1
    elif any(map(sentence.__contains__, class_two)):
        return 2
    elif any(map(sentence.__contains__, class_three)):
        return 3
    else:
        return 4