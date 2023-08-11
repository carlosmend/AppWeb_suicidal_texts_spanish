import pickle
from tensorflow.keras.models import load_model
from collections import Counter
#---------------------------------------------------------
#predictive models
with open('predictive_models/svm_poly.sav', 'rb') as f:
    poly= pickle.load(f)

with open('predictive_models/svm_rbf.sav', 'rb') as f:
    rbf= pickle.load(f)

with open('predictive_models/svm_sigmoid.sav', 'rb') as f:
    sigmoid= pickle.load(f)

with open('predictive_models/lr_saga.sav', 'rb') as f:
    saga= pickle.load(f)

with open('predictive_models/lr_lbfgs.sav', 'rb') as f:
    lbfgs= pickle.load(f)

cnn = load_model('predictive_models/cnn.h5')

#TF-IDF models
#-----------------------------------------
with open('predictive_models/tfidf.pkl', 'rb') as f:
    tfidf1= pickle.load(f)

with open('predictive_models/tfidf2.pkl', 'rb') as f:
    tfidf2= pickle.load(f)

with open('predictive_models/tfidf3.pkl', 'rb') as f:
    tfidf3= pickle.load(f)


#----------------------------------------------------------------
def predict(text):
    predictions=[]
    text_poly  = tfidf1.transform([text]).toarray()
    text_rbf = tfidf2.transform([text]).toarray()
    text_sigmoid = tfidf3.transform([text]).toarray()
    text_cnn= tfidf2.transform([text]).toarray()
    text_sag = tfidf3.transform([text]).toarray()
    text_lbfgs = tfidf3.transform([text]).toarray()
    
    #change of weights in few words
    index_1= tfidf1.vocabulary_["vida"]
    index_2=tfidf2.vocabulary_["vida"]
    index_3=tfidf3.vocabulary_["vida"]
    
    text_sigmoid [:, index_3] = text_sigmoid [:, index_3]*0.32
    text_rbf  [:, index_2] =text_rbf [:, index_2]*0.11
    text_poly [:, index_1] = text_poly [:, index_1]*0.45
    text_cnn  [:, index_2] =text_cnn [:, index_2]*0.4
    text_sag [:, index_3] =text_sag [:, index_3]*0.28
    text_lbfgs  [:, index_3] =text_lbfgs [:, index_3]*0.32
   
    index_1= tfidf1.vocabulary_["acabar"]
    index_2=tfidf2.vocabulary_["acabar"]
    index_3=tfidf3.vocabulary_["acabar"]
        
    text_sigmoid [:, index_3] = text_sigmoid [:, index_3]*0
    text_rbf  [:, index_2] =text_rbf [:, index_2]*0
    text_poly [:, index_1] = text_poly [:, index_1]*-2
    text_cnn  [:, index_2] =text_cnn [:, index_2]*-1
    text_sag [:, index_3] =text_sag [:, index_3]*-7
    text_lbfgs  [:, index_3] =text_lbfgs [:, index_3]*-7

    index_1= tfidf1.vocabulary_["infierno"]
    index_2=tfidf2.vocabulary_["infierno"]
    index_3=tfidf3.vocabulary_["infierno"]

    text_sigmoid [:, index_3] = text_sigmoid [:, index_3]*4.5
    text_rbf  [:, index_2] =text_rbf [:, index_2]*0
    text_poly [:, index_1] = text_poly [:, index_1]*1.5
    text_cnn  [:, index_2] =text_cnn [:, index_2]*1.5
    text_sag [:, index_3] =text_sag [:, index_3]*1.5
    text_lbfgs  [:, index_3] =text_lbfgs [:, index_3]*1.7

    index_1= tfidf1.vocabulary_["renunciar"]
    index_2=tfidf2.vocabulary_["renunciar"]
    index_3=tfidf3.vocabulary_["renunciar"]
        
    text_sigmoid [:, index_3] = text_sigmoid [:, index_3]*3
    text_rbf  [:, index_2] =text_rbf [:, index_2]*0
    text_poly [:, index_1] = text_poly [:, index_1]*4
    text_cnn  [:, index_2] =text_cnn [:, index_2]*4
    text_sag [:, index_3] =text_sag [:, index_3]*0
    text_lbfgs  [:, index_3] =text_lbfgs [:, index_3]*5

    predictions.append(float("{0:.7f}".format(sigmoid.predict_proba(text_sigmoid)[0][1])))
    predictions.append(float("{0:.7f}".format(rbf.predict_proba(text_rbf)[0][1])))
    predictions.append(float("{0:.7f}".format(poly.predict_proba(text_poly)[0][1])))
    predictions.append(float("{0:.7f}".format(saga.predict_proba(text_sag)[0][1])))
    predictions.append(float("{0:.7f}".format(lbfgs.predict_proba(text_lbfgs)[0][1])))
    predictions.append(float("{0:.7f}".format(cnn.predict(text_cnn,verbose=0)[0][0])))
    

    list_pred_class=['suicida' if(isinstance(x,float) and x>=0.5) else 'no_suicida' for x in predictions] 
    list_names=['SVM_sigmoid','SVM_rbf','SVM_poly','LR_sag','LR_lbfgs','CNN']
    moda=Counter(list_pred_class).most_common()[0][0]
    prob_class = list(zip(list_names,predictions,list_pred_class))

    return prob_class,moda

