#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 21:52:43 2022

@author: subhabrata
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle

dta = pd.read_csv("/home/subhabrata/Downloads/spamhamdata.csv")
mail_data = dta.where((pd.notnull(dta)),'')
mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1
x = mail_data['Massage']
y = mail_data['Category']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, random_state= 7)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english',lowercase=True)
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

model = LinearSVC()
model.fit(x_train_features, y_train)

pickle.dump(model,open("SpamOrHam.pkl",'wb'))
pickle.dump(feature_extraction, open("Vectorizer.pkl",'wb'))