#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 22:03:07 2022

@author: subhabrata
"""

import streamlit as st
import pickle

model = pickle.load(open('SpamOrHam.pkl','rb'))
vectorizer = pickle.load(open('Vectorizer.pkl','rb'))


def SpamOrHam(Mail):
    mail_features = vectorizer.transform([Mail])
    pred = model.predict(mail_features)
    return pred

def main():
    st.title("Spam or Ham")
    html_temp = """
    <div style = "background-color:#025246; padding:10px">
    <h2 style = "color:white;text-align:center;">Detect Spam Mail Using SVM </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    Mail = st.text_input("Mail", "Type Here")
    ham_html='''
    <div style = "background-color:#F4D03F; padding:10px">
    <h2 style = "color:white;text-align:center;"> This is not a Spam </h2>
    </div>
    '''
    
    spam_html='''
    <div style = "background-color:#F08080; padding:10px">
    <h2 style = "color:black;text-align:center;"> This is a Spam </h2>
    </div>
    '''
    
    if st.button("Check"):
        output = SpamOrHam(Mail)
        if(output[0]==0):
            st.markdown(spam_html, unsafe_allow_html=True)
        else:
            st.markdown(ham_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main()