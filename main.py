# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
"""
Today is December 21 
Machine Learning Project for Course project of Machine Learning and Deep Learning
This is my first try to do a complete python project as a Data Science Student 
"""
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

def cleanend_data(text):
    # Use a breakpoint in the code line below to debug your script.
    text = text.lower()
    text = re.sub('#><=', '', text)
    text = re.sub('[0-9]+', '', text)
    text = re.sub('[\,]', '', text)
    text = re.sub('[\#]', '', text)
    text = re.sub('[\'\"]', '', text)
    text = re.sub('[\.\$\@\!\*\&\?\_\__]', '', text)
    return text

if __name__ == '__main__':

    path = "./nlp-getting-started/train.csv"
    print(path)
    raw_data = pd.read_csv(path)
    #print(raw_data.head(10))
    raw_data['text_clean'] = raw_data.text.map(cleanend_data)
    stop_words =frozenset(["'tis", "tis", "a", "amp", "30th", "22", "ye", "I", "you", "20th", "me", "get", "st",
                    "sráid", "ll", "le", "tú", "bhí", "faithfully", "tír"])
    del raw_data['text']
    del raw_data['id']
    del raw_data['location']
    del raw_data['keyword']
    y_data=raw_data['target'].copy()
    del raw_data['target']
    X_train, X_val, y_train, y_val = train_test_split(raw_data, y_data, train_size=0.75, random_state=123)
    cv = CountVectorizer(stop_words="english",min_df=3)
    X = cv.fit_transform(X_train.text_clean)
    column_name=cv.get_feature_names_out()
    X_tr_vec=pd.DataFrame(X.toarray(),columns=column_name)
    X_val = cv.transform(X_val.text_clean)
    column_name = cv.get_feature_names_out()
    X_val_vec=pd.DataFrame(X_val.toarray(),columns=column_name)
    #********************************************* Simple logestic regression ****************************************************
    classmodel = LogisticRegression()
    classmodel.fit(X_tr_vec, y_train)
    y_train_pred_lr = classmodel.predict(X_tr_vec)
    y_val_pred_lr = classmodel.predict(X_val_vec)
    tr_acc = accuracy_score(y_train, y_train_pred_lr)
    val_acc = accuracy_score(y_val, y_val_pred_lr)
    print("the accuracy score of the train is : {}, and the accuracy score of validation is : {}".format(tr_acc,val_acc))
    #************************************************** train data ******************************************************************
    path = "./nlp-getting-started/test.csv"
    test_data = pd.read_csv(path)
   # print(test_data.head(10))
    test_data['text_clean'] = test_data.text.map(cleanend_data)
    del test_data['text']
    del test_data['location']
    del test_data['keyword']
    out_data=test_data.copy()
    del out_data["text_clean"]
    del test_data['id']
    #print(test_data.columns)
    #y_test_data=test_data['target'].copy()
    #del test_data['target']
    test_data = cv.transform(test_data.text_clean)
    column_name = cv.get_feature_names_out()
    X_test_vec=pd.DataFrame(test_data.toarray(),columns=column_name)
    out_data["target"] = classmodel.predict(X_test_vec)
    #test_acc = accuracy_score(y_test_data, y_test_pred_lr)
    #print("the accuracy score of the test is : {}".format(test_acc))
    print(out_data.head(10))
    out_data.to_csv('./nlp-getting-started/predict.csv', index=False)
