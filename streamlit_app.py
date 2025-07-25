import streamlit as st
import pandas as pd
import pickle
import gzip
import requests

st.title('🪙 Loan Eligibility App')

st.info('This apps uses machine learning model to check loan eligibility')

with st.expander('Data'):
  st.write('**Raw Data**')
  url = "https://raw.githubusercontent.com/Nijesu-alt/njmachinelearning/refs/heads/master/loan_approval_dataset.csv"
  df = pd.read_csv(url)
  df

with st.sidebar:
  st.sidebar.markdown("<h3 style='color:#F63366;'>Customer Profile</h3>", unsafe_allow_html=True)
  st.header('Input Features')
  no_of_dep = st.slider('Number of Dependents', 0, 10, 5)
  income = st.number_input('Annual Income', 100000, 10000000, 100000)
  loan_amount = st.number_input('Loan Amount', 300000, 50000000, 1000000, 100000)
  score = st.slider('Cibil Score', 2, 20, 12)
  res_ass_val = st.number_input('Residential Asset Value', 100000, 40000000, 20000000, 500000)
  com_ass_val = st.number_input('Commercial Asset Value', 0, 20000000, 500000, 500000)
  lux_ass_val = st.number_input('Luxury Asset Value', 200000, 50000000, 1000000, 500000)
  bank_ass_val = st.number_input('Bank Asset Value', 0, 20000000, 500000, 100000)
  
