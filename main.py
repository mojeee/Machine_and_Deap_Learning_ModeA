# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
"""
Today is December 21 
Machine Learning Project for Course project of Machine Learning and Deep Learning
This is my first try to do a complete python project as a Data Science Student
Today is january 13, I add word clound and use SVM
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

    # before everthing we need to assign flag for each part to work easier
    #reading the file
    read_flag=True
    # executing some cleaning on the raw data
    cleaning_flag=True
    # count plot of target value to check that is unbalance or not
    countplot_flag=False
    # plot word cloud to see what word are most frequent in the tweets
    wordcloud_flag=True
    tain_val_split_flag=True
    simple_logestic_flag=True
    gridcv_logestic_flag=False
    simple_SVM_flag=False
    gridcv_SVM_flag=False
    trainData_evaluation_flag=False

    if read_flag:
        path = "./nlp-getting-started/train.csv"
        print(path)
        raw_data = pd.read_csv(path)
        #print(raw_data.head(10))

    if cleaning_flag:
        raw_data['text_clean'] = raw_data.text.map(cleanend_data)
        del raw_data['text']
        del raw_data['id']
        del raw_data['location']
        del raw_data['keyword']

    if countplot_flag:
        sns.set_theme(style="darkgrid")
        sns.countplot(x='target', data=raw_data)
        plt.show()

    if wordcloud_flag:
        all_tweet = " ".join(review for review in raw_data.text_clean)
        mask = np.array(Image.open("./nlp-getting-started/twitter6.jpg"))
        wordcloud = WordCloud(background_color="white", max_words=1000, mask=mask).generate(all_tweet)
        #image_colors = ImageColorGenerator(mask)
        plt.figure(figsize=[30,30])
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
        plt.savefig("./nlp-getting-started/wordcloud.png", format="png")

    y_data = raw_data['target'].copy()
    del raw_data['target']

    if tain_val_split_flag:
        X_train, X_val, y_train, y_val = train_test_split(raw_data, y_data, train_size=0.75, random_state=123)
        cv = CountVectorizer(stop_words="english",min_df=4)
        X = cv.fit_transform(X_train.text_clean)
        column_name=cv.get_feature_names_out()
        X_tr_vec=pd.DataFrame(X.toarray(),columns=column_name)
        X_val = cv.transform(X_val.text_clean)
        column_name = cv.get_feature_names_out()
        X_val_vec=pd.DataFrame(X_val.toarray(),columns=column_name)

    if simple_logestic_flag:
        classmodel = LogisticRegression()
        classmodel.fit(X_tr_vec, y_train)
        y_train_pred_lr = classmodel.predict(X_tr_vec)
        y_val_pred_lr = classmodel.predict(X_val_vec)
        tr_acc = accuracy_score(y_train, y_train_pred_lr)
        val_acc = accuracy_score(y_val, y_val_pred_lr)
        print("the accuracy score of the train is : {}, and the accuracy score of validation is : {}".format(tr_acc,val_acc))

    if gridcv_logestic_flag:
        C = [0.001, 0.01, 0.1, 1.,1.5,2, 10, 100]
        for c in C:
            classmodel = LogisticRegression(C=c,max_iter = 200)
            classmodel.fit(X_tr_vec, y_train)
            y_train_pred_lr = classmodel.predict(X_tr_vec)
            y_val_pred_lr = classmodel.predict(X_val_vec)
            tr_acc = accuracy_score(y_train, y_train_pred_lr)
            val_acc = accuracy_score(y_val, y_val_pred_lr)
            print(f"LR. C= {c}.\tTrain ACC: {tr_acc}\tVal Acc: {val_acc}")

    if simple_SVM_flag:
        clf = svm.SVC(kernel='rbf',C=1)
        clf.fit(X_tr_vec, y_train)
        y_train_pred_lr = clf.predict(X_tr_vec)
        y_val_pred_lr = clf.predict(X_val_vec)
        tr_acc = accuracy_score(y_train, y_train_pred_lr)
        val_acc = accuracy_score(y_val, y_val_pred_lr)
        print(f"SVM method \tTrain ACC: {tr_acc}\tVal Acc: {val_acc}")

    if gridcv_SVM_flag:
        lr = svm.SVC(kernel='rbf')
        cv = CountVectorizer(stop_words="english", min_df=3)
        X = cv.fit_transform(raw_data.text_clean)
        column_name = cv.get_feature_names_out()
        X_all = pd.DataFrame(X.toarray(), columns=column_name)
        parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1,1, 10],'gamma': [0.1,1, 10]}
        clf = GridSearchCV(lr, parameters,scoring = "accuracy",cv = 4)
        clf.fit(X_all, y_data)
        print("best Parameter is: {}".format(clf.best_params_))
        print("best score is: {}".format(clf.best_score_))
    '''y_train_pred_lr = clf.predict(X_tr_vec)
    y_val_pred_lr = clf.predict(X_val_vec)
    tr_acc = accuracy_score(y_train, y_train_pred_lr)
    val_acc = accuracy_score(y_val, y_val_pred_lr)
    print(f"SVM method \tTrain ACC: {tr_acc}\tVal Acc: {val_acc}")'''

    if trainData_evaluation_flag:
        path = "./nlp-getting-started/test.csv"
        test_data = pd.read_csv(path)
        test_data['text_clean'] = test_data.text.map(cleanend_data)
        del test_data['text']
        del test_data['location']
        del test_data['keyword']
        out_data=test_data.copy()
        del out_data["text_clean"]
        del test_data['id']
        test_data = cv.transform(test_data.text_clean)
        column_name = cv.get_feature_names_out()
        X_test_vec=pd.DataFrame(test_data.toarray(),columns=column_name)
        out_data["target"] = clf.predict(X_test_vec)
        print(out_data.head(10))
        out_data.to_csv('./nlp-getting-started/predict.csv', index=False)
