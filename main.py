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
import tensorflow #the backend used by Keras (there are different beckend)
from tensorflow.keras.models import Sequential #import the type of mpdel: sequential (e.g., MLP)
from tensorflow.keras.layers import Input, Dense #simple linear layer
from tensorflow.keras.utils import to_categorical # transformation for classification labels
from keras.utils.vis_utils import plot_model
from sklearn.neighbors import KNeighborsClassifier
from keras.wrappers.scikit_learn import KerasClassifier

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
    wordcloud_flag=False
    tain_val_split_flag=True
    simple_logestic_flag=False
    gridcv_logestic_flag=False
    KNN_flag = False
    simple_SVM_flag=True
    gridcv_SVM_flag=False# it doesn't need train_test split
    NeuralNetworkFlag=False# it doesn't need train_test split
    optimization_NN_flag=False # it doesn't need train_test split
    trainData_evaluation_flag=True


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
        clf = LogisticRegression()
        clf.fit(X_tr_vec, y_train)
        y_train_pred_lr = clf.predict(X_tr_vec)
        y_val_pred_lr = clf.predict(X_val_vec)
        tr_acc = accuracy_score(y_train, y_train_pred_lr)
        val_acc = accuracy_score(y_val, y_val_pred_lr)
        print("the accuracy score of the train is : {}, and the accuracy score of validation is : {}".format(tr_acc,val_acc))

    if gridcv_logestic_flag:
        C = [0.001, 0.01, 0.1, 1.,1.5,2, 10]
        tr_acc=[]
        val_acc=[]
        for c in C:
            model = LogisticRegression(C=c,max_iter = 200)
            model.fit(X_tr_vec, y_train)
            y_train_pred_lr = model.predict(X_tr_vec)
            y_val_pred_lr = model.predict(X_val_vec)
            tr_acc.append(accuracy_score(y_train, y_train_pred_lr))
            val_acc.append(accuracy_score(y_val, y_val_pred_lr))
            print(f"LR. C= {c}.\tTrain ACC: {accuracy_score(y_train, y_train_pred_lr)}\tVal Acc: {accuracy_score(y_val, y_val_pred_lr)}")
        fig = plt.figure(figsize=(6, 4))
        plt.plot(C,tr_acc, label="train")
        plt.plot(C,val_acc,label="validation")
        plt.xlabel("Inverse of Regularization parameter")
        plt.ylabel("Accuracy")
        plt.title("GridCV of Logistic Regression")
        plt.legend()
        plt.show()

    if KNN_flag:
        accuracy_values_train = []
        accuracy_values_test = []
        k_values = range(1, 15)
        for k in k_values:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_tr_vec, y_train)
            y_pred_train = model.predict(X_tr_vec)
            y_pred_val = model.predict(X_val_vec)
            accuracy_values_train.append(accuracy_score(y_pred_train, y_train))
            accuracy_values_test.append(accuracy_score(y_pred_val, y_val))
        fig = plt.figure(figsize=(6, 4))
        plt.plot(k_values, accuracy_values_train, label="train")
        plt.plot(k_values, accuracy_values_test, label="test")
        plt.xlabel("K")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    if simple_SVM_flag:
        c=1
        gama=0.1
        clf = svm.SVC(kernel='rbf',C=c)
        clf.fit(X_tr_vec, y_train)
        y_train_pred_lr = clf.predict(X_tr_vec)
        y_val_pred_lr = clf.predict(X_val_vec)
        tr_acc = accuracy_score(y_train, y_train_pred_lr)
        val_acc = accuracy_score(y_val, y_val_pred_lr)
        print(f"SVM method with parameter C: {c}\tTrain ACC: {tr_acc}\tVal Acc: {val_acc}")
        flag_print=False
        if flag_print:
            c_values=[0.1, 1, 10]
            rbf_acc_tr=[0.7644070765,0.915046417936,0.9791557190]
            rbf_acc_val=[0.7326680672,0.813550420168,0.78046218487]
            linear_acc_tr=[0.8535645472,0.91154317743,0.94867752]
            linear_acc_val=[0.81460084033,0.7962184873,0.7463235294]
            fig = plt.figure(figsize=(6, 4))
            plt.plot(c_values, rbf_acc_tr, label="Train accuracy of RBF kernel", color="k", linestyle='dashed', marker='*')
            plt.plot(c_values, rbf_acc_val, label="Validation accuracy of RBF kernel", color="b", marker='*')
            plt.plot(c_values, linear_acc_tr, label="Train accuracy of Linear kernel",color="r", linestyle='dashed', marker='o')
            plt.plot(c_values, linear_acc_val, label="Validation accuracy of Linear kernel", color="g", marker='o')
            plt.xlabel("C")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

    if gridcv_SVM_flag:
        lr = svm.SVC(kernel='rbf')
        cv = CountVectorizer(stop_words="english", min_df=3)
        X = cv.fit_transform(raw_data.text_clean)
        column_name = cv.get_feature_names_out()
        X_all = pd.DataFrame(X.toarray(), columns=column_name)
        parameters = {'kernel': ('linear', 'rbf')}
        #parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1,1, 10],'gamma': [0.1,1, 10]}
        clf = GridSearchCV(lr, parameters,scoring = "accuracy",cv = 4)
        clf.fit(X_all, y_data)
        print("best Parameter is: {}".format(clf.best_params_))
        print("best score is: {}".format(clf.best_score_))
    '''y_train_pred_lr = clf.predict(X_tr_vec)
    y_val_pred_lr = clf.predict(X_val_vec)
    tr_acc = accuracy_score(y_train, y_train_pred_lr)
    val_acc = accuracy_score(y_val, y_val_pred_lr)
    print(f"SVM method \tTrain ACC: {tr_acc}\tVal Acc: {val_acc}")'''
    if NeuralNetworkFlag:

        cv = CountVectorizer(stop_words="english",min_df=4)
        X = cv.fit_transform(raw_data.text_clean)
        column_name=cv.get_feature_names_out()
        X_tr_vec_NN=pd.DataFrame(X.toarray(),columns=column_name)
        feature_vector_length = X_tr_vec_NN.shape[1]
        print(X_tr_vec_NN.shape[1])
        num_classes = 2 # fake or not fake
        y_train_NN = to_categorical(y_data, num_classes)

        clf = Sequential()  # we first define how the "model" looks like
        clf.add(Dense(input_dim=feature_vector_length, units=10, activation='relu'))  # input layer
        clf.add(Dense(units=10, activation='relu'))  # input layer
        clf.add(Dense(num_classes, activation='softmax'))  # output layer
        print(clf.summary())
        #plot_model(model, show_shapes=True)
        # Configure the model and start training
        clf.compile(loss='categorical_crossentropy',  # loss metric
                      optimizer='sgd',  # optimizer
                      metrics=['accuracy'])  # displayed metric
        history=clf.fit(X_tr_vec_NN, y_train_NN, epochs=20, batch_size=4, verbose=1, validation_split=0.25)
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        # plt.ylim(0.8, 1)
        plt.show()
        # summarize history for accuracy
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        # plt.ylim(0.8, 1)
        plt.show()
        # see the testing performance
        #test_results = model.evaluate(X_test, y_test_cat, verbose=1)
        #print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
    if optimization_NN_flag:
        cv = CountVectorizer(stop_words="english",min_df=4)
        X = cv.fit_transform(raw_data.text_clean)
        column_name=cv.get_feature_names_out()
        X_tr_vec_NNOP=pd.DataFrame(X.toarray(),columns=column_name)
        feature_vector_length = X_tr_vec_NNOP.shape[1]
        print(X_tr_vec_NNOP.shape[1])
        num_classes = 2 # fake or not fake
        y_train_NNOP = to_categorical(y_data, num_classes)

        def create_model():
            # create model
            model = Sequential()
            model.add(Dense(input_dim=feature_vector_length, units=10, activation='relu'))  # input layer
            model.add(Dense(units=10, activation='relu'))  # input layer
            model.add(Dense(num_classes, activation='softmax'))  # output layer
            model.compile(loss='categorical_crossentropy',  # loss metric
                          optimizer='sgd',  # optimizer
                          metrics=['accuracy'])  # displayed metric
            return model

        model = KerasClassifier(build_fn=create_model, verbose=0)
        # define the grid search parameters
        batch_size = [4, 8, 16, 20]
        epochs = [5,10, 20]
        param_grid = dict(batch_size=batch_size, epochs=epochs)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
        grid_result = grid.fit(X_tr_vec_NNOP, y_train_NNOP)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

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
        if ~ NeuralNetworkFlag:
            out_data["target"]=clf.predict(X_test_vec)
        print(out_data.head(10))
        out_data.to_csv('./nlp-getting-started/predict.csv', index=False)
