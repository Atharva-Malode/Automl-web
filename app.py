import streamlit as st
import pandas as pd
import numpy as np
import os
from pycaret.classification import *
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from flaml import AutoML
from sklearn.model_selection import train_test_split
import pickle

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.title("AutoML using flaml")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    chose_model = st.selectbox('Type of Problem Your are trying to solve', ['classification', 'regression'])
    chose_time = int(st.text_input('Time Budget', '20'))
    if st.button('Run Modelling'): 
        X = df.drop(columns = [chosen_target])
        y = df[chosen_target]
        st.write('Shape of X:', X.shape)
        st.write('Shape of y:', y.shape)
        df = pd.concat([X,y],axis = 1)
        st.dataframe(df)
        automl = AutoML()
        settings = {
        "time_budget": chose_time,  # total runtime in seconds
        "metric": 'mse',  # primary metrics can be chosen from: ['accuracy','roc_auc','f1','log_loss','mae','mse','r2']
        "task": chose_model,  # task type  classification or regression
         }
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)
        automl.fit(X_train=X_train, y_train=y_train, **settings)
        st.write('The best model to solve Problem is :',automl.best_estimator)
        st.write(automl.best_config)
        st.write('Best accuracy on validation data: {0:.4g}'.format(1-automl.best_loss))

if choice == "Download": 
    with open("automl.pkl", "wb") as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

    with open('automl.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")