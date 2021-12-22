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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    path = "./nlp-getting-started/train.csv"
    print(path)
    raw_data = pd.read_csv(path)

    #print(raw_data.head(10))
    raw_data['text_clean'] = raw_data.text.map(cleanend_data)
    #print(raw_data.head(10))
    #stop_words = frozenset(["i", "a", "the","is","are"])
    stop_words =frozenset(["'tis", "tis", "a", "amp", "30th", "22", "ye", "I", "you", "20th", "me", "get", "st",
                    "sráid", "ll", "le", "tú", "bhí", "faithfully", "tír"])

    #print(raw_data.columns)
    del raw_data['text']
    del raw_data['id']
    del raw_data['location']
    del raw_data['keyword']
    #print(raw_data.columns)
    y_data=raw_data['target'].copy()
    del raw_data['target']
    #print(y_data.head())
    #print(raw_data.head())
    #print(raw_data.shape)
    #print(raw_data.shape)
    X_train, X_val, y_train, y_val = train_test_split(raw_data, y_data, train_size=0.75, random_state=123)
    #print(X_train.shape)
    #print(X_val.shape)
    cv = CountVectorizer(stop_words="english",min_df=3)
    X = cv.fit_transform(X_train.text_clean)
    column_name=cv.get_feature_names_out()
    X_tr_vec=pd.DataFrame(X.toarray(),columns=column_name)
    #print(len(column_name))
    #print(len(X.toarray()))

    X_val = cv.transform(X_val.text_clean)
    column_name = cv.get_feature_names_out()
    X_val_vec=pd.DataFrame(X_val.toarray(),columns=column_name)

    #print((X_tr_vec.shape))
    #print((X_val_vec.shape))
    #print(X_vec.head())
    classmodel = LogisticRegression()
    classmodel.fit(X_tr_vec, y_train)
    y_train_pred_lr = classmodel.predict(X_tr_vec)
    y_val_pred_lr = classmodel.predict(X_val_vec)
    tr_acc = accuracy_score(y_train, y_train_pred_lr)
    val_acc = accuracy_score(y_val, y_val_pred_lr)
    print("the accuracy score of the train is : {}, and the accuracy score of validation is : {}".format(tr_acc,val_acc))



    '''print(nlp_train_data.columns)
    print(nlp_train_data.text.head())
    print(nlp_train_data.text[5])
    print(nlp_train_data.text.shape)
    print(nlp_train_data.target)
    print(nlp_train_data.keyword.dropna().value_counts())
    print(nlp_train_data.target.value_counts())
    '''
    '''
    print(nlp_train_data.shape)
    # make the data a little bit clean
    raw_data = nlp_train_data.copy()
    nlp_train_data['text_clean'] = nlp_train_data.text.map(cleanend_data)
    del nlp_train_data['text']
    # print(nlp_train_data.head())
    y_data = nlp_train_data['target']
    del nlp_train_data['target']

    X_train, X_val, y_train, y_val = train_test_split(nlp_train_data, y_data, train_size=0.75, random_state=123)
    # vectorize the text
    # we can add some stop words here 
    cv = CountVectorizer()
    data_cv_train = cv.fit_transform(X_train.text_clean)
    data_train_dtm = pd.DataFrame(data_cv_train.toarray(), columns=cv.get_feature_names())
    data_train_dtm.index = X_train.index
    data_cv_val = cv.transform(X_val.text_clean)
    data_val_dtm = pd.DataFrame(data_cv_val.toarray(), columns=cv.get_feature_names())
    data_val_dtm.index = X_val.index
    print(data_train_dtm)
    print(nlp_train_data.columns)
    print(data_train_dtm.columns)

    data_dtm_copy = data_train_dtm.copy()
    del data_train_dtm['id']
    del data_train_dtm['location']
    del data_val_dtm['id']
    del data_val_dtm['location']
    print('id' in data_train_dtm.columns)
    print('location' in data_train_dtm.columns)
    print('keyword' in data_train_dtm.columns)
    print('text_clean' in data_train_dtm.columns)
    print('target' in data_train_dtm.columns)
    print(data_train_dtm.shape)

    colsum = data_train_dtm.sum()
    print(colsum.shape)
    print(colsum.describe())
    print(colsum > 10 * colsum.mean())
    stopwords = []
    data_flt = []
    tenthmean = (np.mean(colsum))
    for column in data_train_dtm.columns:
        if data_train_dtm[column].sum() > tenthmean:
            stopwords.append(column)
    data_train_filt = data_train_dtm.loc[:, stopwords]
    data_val_filt = data_val_dtm.loc[:, stopwords]
    print(len(stopwords))
    print(data_train_filt.shape)
    print(data_train_filt.head())

    scaler2 = StandardScaler()
    scaler2.fit(data_train_filt)
    data_train_model = scaler2.transform(data_train_filt)
    data_val_model = scaler2.transform(data_val_filt)

    print(len(y_train))
    print(data_val_filt)
    classmodel = LogisticRegression()
    classmodel.fit(data_train_filt, y_train)
    y_train_pred_lr = classmodel.predict(data_train_model)
    y_val_pred_lr = classmodel.predict(data_val_model)
    '''