import pickle
import pandas as pd
from text_classification.preprocessing import clean_sentence
from text_classification.utils import carbon_class_filter, get_sum_probs, get_majority_pred_soft

# instantiate model paths
LOGREG_VECT = "text_classification/saved_models/model_LR_vectorizer.pkl"
LOGREG_MODEL = "text_classification/saved_models/model_LR.pkl"
SVM_VECT = "text_classification/saved_models/model_SVM_vectorizer.pkl"
SVM_MODEL = "text_classification/saved_models/model_SVM.pkl"
NB_VECT = "text_classification/saved_models/model_NB_vectorizer.pkl"
NB_MODEL = "text_classification/saved_models/model_NB.pkl"
RF_VECT = "text_classification/saved_models/model_RF_vectorizer.pkl"
RF_MODEL = "text_classification/saved_models/model_RF.pkl"
CB_VECT = "text_classification/saved_models/model_CB_vectorizer.pkl"
CB_MODEL = "text_classification/saved_models/model_CB.pkl"

def text_classification_pipeline(df):
    if len(df) == 0:
        return []
    # LOG REG
    lr_vect = pickle.load(open(LOGREG_VECT, "rb"))
    lr_model = pickle.load(open(LOGREG_MODEL, "rb"))
    lr_vected_text = lr_vect.transform(df.cleaned_sentence)
    lr_pred = lr_model.predict_proba(lr_vected_text)
    df['lr_prob_0'] = [i[0] for i in lr_pred]
    df['lr_prob_1'] = [i[1] for i in lr_pred]
    df['lr_prob_2'] = [i[2] for i in lr_pred]
    df['lr_prob_3'] = [i[3] for i in lr_pred]
    df['lr_prob_4'] = [i[4] for i in lr_pred]

    # NB
    nb_vect = pickle.load(open(NB_VECT, "rb"))
    nb_model = pickle.load(open(NB_MODEL, "rb"))
    nb_vected_text = nb_vect.transform(df.cleaned_sentence)
    nb_pred = nb_model.predict_proba(nb_vected_text)
    df['nb_prob_0'] = [i[0] for i in nb_pred]
    df['nb_prob_1'] = [i[1] for i in nb_pred]
    df['nb_prob_2'] = [i[2] for i in nb_pred]
    df['nb_prob_3'] = [i[3] for i in nb_pred]
    df['nb_prob_4'] = [i[4] for i in nb_pred]

    # SVM
    svm_vect = pickle.load(open(SVM_VECT, "rb"))
    svm_model = pickle.load(open(SVM_MODEL, "rb"))
    svm_vected_text = svm_vect.transform(df.sentence)
    svm_pred = svm_model.predict_proba(svm_vected_text)
    df['svm_prob_0'] = [i[0] for i in svm_pred]
    df['svm_prob_1'] = [i[1] for i in svm_pred]
    df['svm_prob_2'] = [i[2] for i in svm_pred]
    df['svm_prob_3'] = [i[3] for i in svm_pred]
    df['svm_prob_4'] = [i[4] for i in svm_pred]


    # RF
    rf_vect = pickle.load(open(RF_VECT, "rb"))
    rf_model = pickle.load(open(RF_MODEL, "rb"))
    rf_vected_text = rf_vect.transform(df.cleaned_sentence)
    rf_pred = rf_model.predict_proba(rf_vected_text)
    df['rf_prob_0'] = [i[0] for i in rf_pred]
    df['rf_prob_1'] = [i[1] for i in rf_pred]
    df['rf_prob_2'] = [i[2] for i in rf_pred]
    df['rf_prob_3'] = [i[3] for i in rf_pred]
    df['rf_prob_4'] = [i[4] for i in rf_pred]

    # CB
    cb_vect = pickle.load(open(CB_VECT, "rb"))
    cb_model = pickle.load(open(CB_MODEL, "rb"))
    cb_vected_text = cb_vect.transform(df.cleaned_sentence)
    cb_pred = cb_model.predict_proba(cb_vected_text)
    df['cb_prob_0'] = [i[0] for i in cb_pred]
    df['cb_prob_1'] = [i[1] for i in cb_pred]
    df['cb_prob_2'] = [i[2] for i in cb_pred]
    df['cb_prob_3'] = [i[3] for i in cb_pred]
    df['cb_prob_4'] = [i[4] for i in cb_pred]

    # WORD HEURISTICS
    heu_preds = list(df.apply(carbon_class_filter, axis=1))

    # GET VOTING CLASSIFIER
    df = get_sum_probs(df, heu_preds)
    model_pred  = get_majority_pred_soft(df)

    return model_pred

