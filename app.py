import streamlit as st
import helper 
import joblib
import pickle
import xgboost as xgb # 1. Import xgboost

model = joblib.load('model.joblib')

# 2. Force the model to use the CPU to stop the CUDA warning
model.set_params(device="cpu") 

q_count_dict = pickle.load(open('q_count_dict.pkl','rb'))

st.header('Quora Duplicate Question Checker')

q1 = st.text_input('Enter the question 1')
q2 = st.text_input('Enter The question 2')

if st.button('Find'):
    query = helper.query_point_creator(q1,q2)
    result = model.predict(query)[0]

    if(result == 1):
        st.header('Duplicate')
    else :
        st.header('Not Duplicate')
