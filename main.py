# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
"""
Today is December 21 
Machine Learning Project for Course project of Machine Learning and Deep Learning
This is my first try to do a complete python project as a Data Science Student 
"""
import inline as inline
import matplotlib
import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from PIL import Image


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

def cleanend_data(text):
    # Use a breakpoint in the code line below to debug your script.
    text = text.lower()
    re.sub(r'http\S+', '', text)

    text_tokens = word_tokenize(text)
    all_stopwords = stopwords.words('english')
    more_stop=["https","tco","via","s","rt","st","w","im","re","m","d","v","a","b","c","e","f",
               "g","h","lol","l","n","o","p","k","q","r","s","t","u","v","w","ll","ve","tco","nt"]
    all_stopwords.extend(more_stop)
    tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]
    text = (" ").join(tokens_without_sw)

    text = re.sub('#><=', '', text)
    text = re.sub('[0-9]+', '', text)
    text = re.sub('[\,]', '', text)
    text = re.sub('[\#]', '', text)
    text = re.sub('[\'\"]', '', text)
    text = re.sub('[\.\$\@\!\*\&\?\_\__]', '', text)


    '''stop_words =["'tis", "tis", "a", "amp", "30th", "22", "ye", "I", "you", "20th", "me", "get", "st",
                    "sráid", "ll", "le", "tú", "bhí", "faithfully", "tír","https","lol","im"]
    for a in stop_words:
        text=text.replace(a, "")'''
    return text

if __name__ == '__main__':

    path = "./nlp-getting-started/train.csv"
    print(path)
    raw_data = pd.read_csv(path)
    #print(raw_data.head(10))
    raw_data['text_clean'] = raw_data.text.map(cleanend_data)

    del raw_data['text']
    del raw_data['id']
    del raw_data['location']
    del raw_data['keyword']
    sns.countplot(x='target', data=raw_data)
    plt.show()
    #**** word cloud
    #df = pd.DataFrame(raw_data[['text_clean', 'Tweet']])

    Df_critical = raw_data[raw_data['target'] == 1]
    Df_non_critical= raw_data[raw_data['target'] == 0]

    all_tweet = " ".join(review for review in raw_data.text_clean)

    mask = np.array(Image.open("./nlp-getting-started/twitter6.jpg"))
    wordcloud = WordCloud(background_color="white", max_words=1000, mask=mask).generate(all_tweet)
    #image_colors = ImageColorGenerator(mask)
    plt.figure(figsize=[40,40])
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    #wordcloud_ALL = WordCloud(background_color="white", max_words=1000, mask=mask).generate(all_tweet)
    #plt.imshow(wordcloud_ALL, interpolation='bilinear')
    # create coloring from image
    #plt.figure(figsize=[7, 7])
    #plt.imshow(wordcloud_ALL.recolor(color_func=image_colors), interpolation="bilinear")
    #plt.axis("off")
#*************************
    # store to file
    #plt.savefig("./nlp-getting-started/ita_wine.png", format="png")
    #plt.show()
    y_data=raw_data['target'].copy()
    del raw_data['target']
    X_train, X_val, y_train, y_val = train_test_split(raw_data, y_data, train_size=0.75, random_state=123)
    cv = CountVectorizer(stop_words="english",min_df=4)
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
    C = [0.001, 0.01, 0.1, 1.,1.5,2, 10, 100]
    for c in C:
        classmodel = LogisticRegression(C=c,max_iter = 200)
        classmodel.fit(X_tr_vec, y_train)
        y_train_pred_lr = classmodel.predict(X_tr_vec)
        y_val_pred_lr = classmodel.predict(X_val_vec)
        tr_acc = accuracy_score(y_train, y_train_pred_lr)
        val_acc = accuracy_score(y_val, y_val_pred_lr)

        print(f"LR. C= {c}.\tTrain ACC: {tr_acc}\tVal Acc: {val_acc}")
    #*************************************** SVM Method *************************************************
    clf = svm.SVC(kernel='rbf',C=1)
    clf.fit(X_tr_vec, y_train)
    y_train_pred_lr = clf.predict(X_tr_vec)
    y_val_pred_lr = clf.predict(X_val_vec)
    tr_acc = accuracy_score(y_train, y_train_pred_lr)
    val_acc = accuracy_score(y_val, y_val_pred_lr)
    print(f"SVM method \tTrain ACC: {tr_acc}\tVal Acc: {val_acc}")

    #**************************************************Gridcv for SVM*************************************
    '''lr = svm.SVC(kernel='rbf')
    cv = CountVectorizer(stop_words="english", min_df=3)
    X = cv.fit_transform(raw_data.text_clean)
    column_name = cv.get_feature_names_out()
    X_all = pd.DataFrame(X.toarray(), columns=column_name)
    parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1,1, 10],'gamma': [0.1,1, 10]}
    clf = GridSearchCV(lr, parameters,scoring = "accuracy",cv = 4)
    clf.fit(X_all, y_data)
    print("best Parameter is: {}".format(clf.best_params_))
    print("best score is: {}".format(clf.best_score_))'''
    '''y_train_pred_lr = clf.predict(X_tr_vec)
    y_val_pred_lr = clf.predict(X_val_vec)
    tr_acc = accuracy_score(y_train, y_train_pred_lr)
    val_acc = accuracy_score(y_val, y_val_pred_lr)
    print(f"SVM method \tTrain ACC: {tr_acc}\tVal Acc: {val_acc}")'''
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
    out_data["target"] = clf.predict(X_test_vec)
    #test_acc = accuracy_score(y_test_data, y_test_pred_lr)
    #print("the accuracy score of the test is : {}".format(test_acc))
    print(out_data.head(10))
    out_data.to_csv('./nlp-getting-started/predict.csv', index=False)
