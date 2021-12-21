# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
"""
Today is December 21 
Machine Learning Project for Course project of Machine Learning and Deep Learning
This is my first try to do a complete python project as a Data Science Student 
"""
import pandas as pd
import numpy as np
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print('Hi '+name)  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    path= "./nlp-getting-started/train.csv"
    print(path)
    nlp_train_data=pd.read_csv(path)
    print(nlp_train_data.head(10))
    print(nlp_train_data.columns)
    print(nlp_train_data.text.head())
    print(nlp_train_data.text[5])
    print(nlp_train_data.text.shape)
