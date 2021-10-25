import pickle
import pandas as pd
import nltk

# instantiate models
DATA_FOLDER = "../data/"
LOGREG_VECT = DATA_FOLDER + "saved_models/model_logreg_vectorizer.pkl"
LOGREG_MODEL = DATA_FOLDER + "saved_models/model_logreg.pkl"
SVM_VECT = DATA_FOLDER + "saved_models/model_SVM_vectorizer.pkl"
SVM_MODEL = DATA_FOLDER + "saved_models/model_SVM.pkl"
NB_VECT = DATA_FOLDER + "saved_models/model_NB_vectorizer.pkl"
NB_MODEL = DATA_FOLDER + "saved_models/model_NB.pkl"
RF_VECT = DATA_FOLDER + "saved_models/model_RF_vectorizer.pkl"
RF_MODEL = DATA_FOLDER + "saved_models/model_RF.pkl"
META_MODEL = DATA_FOLDER + "modelling/saved_models/model_meta.pkl"


# input : json
# output : json with predictions
def base_modelling_pipeline(processed_csv, prediction_csv):
    '''
    Reads processed data and outputs csv of predictions for each model
    '''
    # READ PROCESSED DATA
    processed_df = pd.read_csv(processed_csv)
    
    # LOGISTIC REGRESSION PREDICTION
    lr_vectorizer = pickle.load(open(LOGREG_VECT, "rb"))
    lr_model = pickle.load(open(LOGREG_MODEL, "rb"))
    lr_transformed_text = lr_vectorizer.transform(processed_df.phrase_stem_emoticon_unique)
    lr_predictions = lr_model.predict_proba(lr_transformed_text)
    processed_df["logreg_prob_pos"] = lr_predictions[:, 2]
    processed_df["logreg_prob_neg"] = lr_predictions[:, 0]

    # SUPPORT VECTOR MACHINE PREDICTION
    svm_vectorizer = pickle.load(open(SVM_VECT, "rb"))
    svm_model = pickle.load(open(SVM_MODEL, "rb"))
    svm_transformed_text = svm_vectorizer.transform(processed_df.phrase_emoticon_generic)
    svm_predictions = svm_model.predict_proba(svm_transformed_text)
    processed_df["SVM_prob_pos"] = svm_predictions[:, 2]
    processed_df["SVM_prob_neg"] = svm_predictions[:, 0]

    # NAIVE BAYES PREDICTION
    nb_vectorizer = pickle.load(open(NB_VECT, "rb"))
    nb_model = pickle.load(open(NB_MODEL, "rb"))
    nb_transformed_text = nb_vectorizer.transform(processed_df.phrase_stem_emoticon_generic)
    nb_predictions = nb_model.predict_proba(nb_transformed_text)
    processed_df["NB_prob_pos"] = nb_predictions[:, 2]
    processed_df["NB_prob_neg"] = nb_predictions[:, 0]

    # RANDOM FOREST PREDICTION
    rf_vectorizer = pickle.load(open(RF_VECT, "rb"))
    rf_model = pickle.load(open(RF_MODEL, "rb"))
    rf_transformed_text = rf_vectorizer.transform(processed_df.phrase_stem_emoticon_generic)
    rf_predictions = rf_model.predict_proba(rf_transformed_text)
    processed_df["RF_prob_pos"] = rf_predictions[:, 2]
    processed_df["RF_prob_neg"] = rf_predictions[:, 0]

    print("LR, SVM, NB, RF predictions complete")

    # FASTTEXT PREDICTION
    fasttext_model = fasttext.load_model(FASTTEXT_MODEL)
    fasttext_df = processed_df.copy()
    # get raw output (('__label__pos', '__label__zer', '__label__neg'), array([0.74627936, 0.19218659, 0.06156404]))
    fasttext_df['raw_output'] = fasttext_df.apply(lambda x: fasttext_model.predict(x['phrase_stem'].replace("\n", ""), k=-1), axis=1)
    # get raw prob [0.74627936, 0.19218659, 0.06156404]
    fasttext_df['raw_prob'] = fasttext_df.apply(lambda x: list(x.raw_output[1]), axis=1)
    # get pos and neg index
    fasttext_df['pos_index'] = fasttext_df.apply(lambda x: fasttext_get_index(list(x.raw_output[0]), 'pos'), axis=1)
    fasttext_df['neg_index'] = fasttext_df.apply(lambda x: fasttext_get_index(list(x.raw_output[0]), 'neg'), axis=1)
    # get prob_pos and prob_neg
    fasttext_df['fasttext_prob_pos'] = fasttext_df.apply(lambda x: x.raw_prob[x.pos_index], axis=1)
    fasttext_df['fasttext_prob_neg'] = fasttext_df.apply(lambda x: x.raw_prob[x.neg_index], axis=1)
    # add to processed_df
    processed_df["fasttext_prob_pos"] = fasttext_df['fasttext_prob_pos']
    processed_df["fasttext_prob_neg"] = fasttext_df['fasttext_prob_neg']

    print("Fasttext predictions complete")

    # BERT PREDICTION
    bert_model_args = ClassificationArgs(num_train_epochs=2, learning_rate=5e-5)
    bert_model = ClassificationModel(model_type = 'bert', \
                                     model_name = BERT_MODEL, \
                                     args = bert_model_args, use_cuda = False)
    bert_pred, bert_raw_outputs = bert_model.predict(processed_df.phrase)
    # convert raw output to probabilities
    bert_probabilities = softmax(bert_raw_outputs, axis=1)
    processed_df['bert_prob_pos'] = bert_probabilities[:, 1]
    processed_df['bert_prob_neg'] = bert_probabilities[:, 2]

    print("BERT predictions complete")
    
    # VADER PREDICTION
    processed_df[["VADER_prob_pos","VADER_prob_neg"]] = load_VADER_model(processed_df)
    
    processed_df.to_csv(prediction_csv, index=False)
    print("BASELINE PREDICTIONS COMPLETE")


def meta_modelling_pipeline(prediction_csv, ensemble_file):
    '''
    Retrieves predictions for each baseline model and outputs final prediction using meta model

    Parameters:
        prediction_dir (str): directory containing all predictions from baseline model
        ensemble_file (str):  filename to save ensemble predictions in
    Return: none
    '''
    # read baseline model predictions 
    predictions_df = pd.read_csv(prediction_csv)
    
    # generate ensemble predictions
    meta_model = pickle.load(open(META_MODEL, "rb"))

    # fit model
    predictions = meta_model.predict_proba(predictions_df[['bert_prob_pos', 'bert_prob_neg', 'fasttext_prob_pos',
       'fasttext_prob_neg', 'logreg_prob_pos', 'logreg_prob_neg',
       'NB_prob_pos', 'NB_prob_neg', 'RF_prob_pos', 'RF_prob_neg',
       'SVM_prob_pos', 'SVM_prob_neg', 'VADER_prob_pos', 'VADER_prob_neg']])[:, 0:3]
    predictions_df["prob_neg"] = 0
    predictions_df["prob_neu"] = 0
    predictions_df["prob_pos"] = 0
    predictions_df[["prob_neg","prob_neu","prob_pos"]] = predictions
    
    ensemble_predictions = pd.DataFrame(data=predictions_df,columns=["restaurant_code", "review_title", "review_body", "review_title_raw", "review_body_raw", "review_date", "account_name", 
        "account_id",  "account_level", "account_photo", "review_photo", "scraped_date", "location", "aspect", 
        "prob_pos", "prob_neu", "prob_neg"]) 
    
    # save ensemble predictions
    ensemble_predictions.to_csv(ensemble_file, index=False)
    print("ENSEMBLE PREDICTIONS COMPLETE")
  