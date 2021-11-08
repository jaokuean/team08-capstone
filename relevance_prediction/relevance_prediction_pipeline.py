# import packages
import pickle
import pandas as pd
import nltk
from relevance_prediction.preprocessing import * # import clean function
import json


# load pretrained models and instantiate them
DATA_FOLDER = "data/"
LOGREG_VECT = DATA_FOLDER + "saved_models/relevance_models/model_LR_vectorizer.pkl"
LOGREG_MODEL = DATA_FOLDER + "saved_models/relevance_models/model_LR.pkl"
SVM_VECT = DATA_FOLDER + "saved_models/relevance_models/model_SVM_vectorizer.pkl"
SVM_MODEL = DATA_FOLDER + "saved_models/relevance_models/model_SVM.pkl"
NB_VECT = DATA_FOLDER + "saved_models/relevance_models/model_NB_vectorizer.pkl"
NB_MODEL = DATA_FOLDER + "saved_models/relevance_models/model_NB.pkl"
RF_VECT = DATA_FOLDER + "saved_models/relevance_models/model_RF_vectorizer.pkl"
RF_MODEL = DATA_FOLDER + "saved_models/relevance_models/model_RF.pkl"
LOGREG_MODEL_BERT = DATA_FOLDER + "saved_models/relevance_models/model_LR_BERT.pkl"
META_MODEL = DATA_FOLDER + "saved_models/relevance_models/model_meta.pkl"

def modelling_pipeline(company):
    """
    Main function that runs pre-trained machine learning models to predict relevance of a sentence.

    Parameters
    ----------
    company : dict of {str : str or dict}
        Dictionary containing a company's name, year, highly relevant sentences and their BERT word embeddings.

    Return
    ------
    ensemble_predictions : dataframe
        Dataframe that contains sentences predicted as relevant. Columns include page number of the sentences, actual sentence text and the probability scores predicted.
    """   
    
    processed_json = company
    sentences_direct = processed_json["bert_relevant_sentences_direct_original"]
    sentences_indirect = processed_json["bert_relevant_sentences_indirect_original"]
    sentences = dict(list(sentences_direct.items()) + list(sentences_indirect.items()))
    
    # obtain sentence embeddings for BERT models
    sentences_direct_embeddings = company["bert_relevant_sentences_direct_original_embeddings"]
    sentences_indirect_embeddings  = company["bert_relevant_sentences_indirect_original_embeddings"]
    sentences_embeddings  = dict(list(sentences_direct_embeddings.items()) + list(sentences_indirect_embeddings.items()))

    page = []
    sentence = []
    sentence_embeddings = []
    
    for page_number,page_sentences in sentences.items():
        for sent in page_sentences:
            page.append(page_number)
            sentence.append(sent)

    for page_number,page_sentences in sentences_embeddings.items():
        for sent in page_sentences:
            sentence_embeddings.append(sent)
    
    processed_df = pd.DataFrame({"page":page,"sentence":sentence,"sentence_embeddings":sentence_embeddings})
    processed_df['cleaned_sentence'] = processed_df['sentence'].apply(clean_sentence)
    
    # MAKE PREDICTIONS
    # LOGISTIC REGRESSION PREDICTION
    lr_vectorizer = pickle.load(open(LOGREG_VECT, "rb"))
    lr_model = pickle.load(open(LOGREG_MODEL, "rb"))
    lr_transformed_text = lr_vectorizer.transform(processed_df.cleaned_sentence)
    lr_predictions = lr_model.predict_proba(lr_transformed_text)
    processed_df["LR_prob_1"] = lr_predictions[:, 1]
    processed_df["LR_prob_0"] = lr_predictions[:, 0]

    # SUPPORT VECTOR MACHINE PREDICTION
    svm_vectorizer = pickle.load(open(SVM_VECT, "rb"))
    svm_model = pickle.load(open(SVM_MODEL, "rb"))
    svm_transformed_text = svm_vectorizer.transform(processed_df.cleaned_sentence)
    svm_predictions = svm_model.predict_proba(svm_transformed_text)
    processed_df["SVM_prob_1"] = svm_predictions[:, 1]
    processed_df["SVM_prob_0"] = svm_predictions[:, 0]

    # NAIVE BAYES PREDICTION
    nb_vectorizer = pickle.load(open(NB_VECT, "rb"))
    nb_model = pickle.load(open(NB_MODEL, "rb"))
    nb_transformed_text = nb_vectorizer.transform(processed_df.cleaned_sentence)
    nb_predictions = nb_model.predict_proba(nb_transformed_text)
    processed_df["NB_prob_1"] = nb_predictions[:, 1]
    processed_df["NB_prob_0"] = nb_predictions[:, 0]

    # RANDOM FOREST PREDICTION
    rf_vectorizer = pickle.load(open(RF_VECT, "rb"))
    rf_model = pickle.load(open(RF_MODEL, "rb"))
    rf_transformed_text = rf_vectorizer.transform(processed_df.cleaned_sentence)
    rf_predictions = rf_model.predict_proba(rf_transformed_text)
    processed_df["RF_prob_1"] = rf_predictions[:, 1]
    processed_df["RF_prob_0"] = rf_predictions[:, 0]

    print("LR, SVM, NB, RF predictions complete")
    
    # LOGISTIC REGRESSION BERT PREDICTION
    lr_bert_model = pickle.load(open(LOGREG_MODEL_BERT, "rb"))
    lr_bert_predictions = lr_bert_model.predict_proba(list(processed_df.sentence_embeddings))
    processed_df["LR_BERT_prob_1"] =  lr_predictions[:, 1]
    processed_df["LR_BERT_prob_0"] =  lr_predictions[:, 0]   

    print("BASELINE PREDICTIONS COMPLETE")        

    
    # STACKING MODEL
    # read baseline model predictions 
    predictions_df = processed_df
    
    # generate ensemble predictions
    meta_model = pickle.load(open(META_MODEL, "rb"))

    # fit model
    predictions = meta_model.predict_proba(predictions_df[['RF_prob_0','RF_prob_1','LR_prob_0', 'LR_prob_1',
       'NB_prob_0','NB_prob_1',"LR_BERT_prob_0","LR_BERT_prob_1",'SVM_prob_0', 'SVM_prob_1']])[:, 0:2]
    predictions_df["prob_0"] = 0
    predictions_df["relevance_prob"] = 0 #prob_1
    predictions_df["relevance"] = meta_model.predict(predictions_df[['RF_prob_0','RF_prob_1','LR_prob_0', 'LR_prob_1',
       'NB_prob_0','NB_prob_1',"LR_BERT_prob_0","LR_BERT_prob_1",'SVM_prob_0', 'SVM_prob_1']])
    predictions_df[["prob_0","relevance_prob"]] = predictions

    predictions_df = predictions_df.loc[predictions_df.relevance == 1]
    ensemble_predictions = pd.DataFrame(data=predictions_df,columns=["page", "sentence", "relevance_prob"]) 

    print("ENSEMBLE PREDICTIONS COMPLETE")        
    
    # save ensemble predictions
    return ensemble_predictions


def relevance_prediction(file_path):
    """
    Main relevance modelling prediction pipeline function.

    Parameters
    ----------
    file_path : str
        String of path to a json file containing company's report details, filtered text and bert emebddings for highly relevant sentences.

    Return
    ------
    output_path : str
        String of path to output file containing "text_output" field which includes pages of relevance sentences, relevant sentences predicted by the model and probability scores.
    """  
    
    with open(file_path, 'r') as infile:
        company = json.load(infile)
    
    company_details = {}
    company_details["company"] = company["company"]
    company_details["year"] = company["year"]
    company_details["url"] = company["url"]

    sentences_direct = company["bert_relevant_sentences_direct_original"]
    sentences_indirect = company["bert_relevant_sentences_indirect_original"]
    sentences = dict(list(sentences_direct.items()) + list(sentences_indirect.items()))
    
    # if there are relevant sentences after filtering by BERT, predict the relevance of the sentences
    if len(sentences) != 0:
        print("RUNNING RELEVANCE MODELLING PIPELINE")
        data = modelling_pipeline(company)
        data_json = json.loads(data.reset_index().drop("index",axis=1).to_json())
        for k,v in data_json.items():
            data_json[k] = list(data_json[k].values())
        company_details["text_output"] = data_json
    else:
        print("NO RELEVANT SENTENCES")
        company_details["text_output"] = data.reset_index().drop("index",axis=1).to_json()
    
    output_path = file_path[:-21] + "_text_output.json"
    
    with open(output_path, "w") as outfile:  
        json.dump(company_details, outfile)
    
    print("DONE RUNNING RELEVANCE MODELLING PIPELINE")
    return output_path



####### PIPELINE ###############
#text_output_path = relevance_prediction(file_path)
