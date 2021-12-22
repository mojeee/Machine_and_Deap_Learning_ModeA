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
import numpy as np
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

def cleanend_data(text):
    # Use a breakpoint in the code line below to debug your script.
    text = text.lower()
    text = re.sub('#><=', '', text)
    text = re.sub('[0-9]+', '', text)
    text = re.sub('[\,]', '', text)
    text = re.sub('[\#]', '', text)
    text = re.sub('[\'\"]', '', text)
    text = re.sub('[\.\$\@\!\*\&\?]', '', text)



    return text



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    path= "./nlp-getting-started/train.csv"
    print(path)
    nlp_train_data=pd.read_csv(path)
    '''
    print(nlp_train_data.head(10))
    print(nlp_train_data.columns)
    print(nlp_train_data.text.head())
    print(nlp_train_data.text[5])
    print(nlp_train_data.text.shape)
    print(nlp_train_data.target)
    print(nlp_train_data.keyword.dropna().value_counts())
    print(nlp_train_data.target.value_counts())
    '''
    print(nlp_train_data.shape)
    #make the data a little bit clean
    raw_data=nlp_train_data.copy()
    nlp_train_data['text_clean']=nlp_train_data.text.map(cleanend_data)
    del nlp_train_data['text']
    #print(nlp_train_data.head())

    #vectorize the text
    # we can add some stop words here 
    cv = CountVectorizer()
    data_cv = cv.fit_transform(nlp_train_data.text_clean)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = nlp_train_data.index
    print(data_dtm)
    print(nlp_train_data.columns)
    print(data_dtm.columns)

    data_dtm_copy=data_dtm.copy()
    del data_dtm['id']
    del data_dtm['location']
    print('id' in data_dtm.columns)
    print('location' in data_dtm.columns)
    print('keyword' in data_dtm.columns)
    print('text_clean' in data_dtm.columns)
    print('target' in data_dtm.columns)
    print(data_dtm.shape)
    y=data_dtm['target']
    del data_dtm['target']
    colsum=data_dtm.sum()
    print(colsum.shape)
    print(colsum.describe())
    print(colsum>10*colsum.mean())
    stopwords=[]
    data_flt=[]
    tenthmean=(colsum.mean())
    for column in data_dtm.columns:
        if data_dtm[column].sum()>tenthmean:
            stopwords.append(column)
    data_filt=data_dtm.loc[:,stopwords]
    print(len(stopwords))
    print(data_filt.shape)
    print(data_filt.head())


