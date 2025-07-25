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
  loan_term = st.slider('Loan Term', 2, 40, 25)
  score = st.slider('Cibil Score', 2, 20, 12)
  res_ass_val = st.number_input('Residential Asset Value', 100000, 40000000, 20000000, 500000)
  com_ass_val = st.number_input('Commercial Asset Value', 0, 20000000, 500000, 500000)
  lux_ass_val = st.number_input('Luxury Asset Value', 200000, 50000000, 1000000, 500000)
  bank_ass_val = st.number_input('Bank Asset Value', 0, 20000000, 500000, 100000)
  edu_level = st.selectbox('Education Level', ['Graduate', 'Not Graduate'])
  job = st.selectbox('Self employed?', ['Yes', 'No'])

data = {
  'no_of_dependents' : no_of_dep,
  'income_annum' : income,
  'loan_amount' : loan_amount,
  'loan_term' : loan_term,
  'cibil_score' : score,
  'residential_assets_value' : res_ass_val,
  'commercial_assets_value' : com_ass_val,
  'luxury_assets_value' : lux_ass_val,
  'bank_asset_value' : bank_ass_val,
  'Education' : edu_level,
  'Self_Employed?' : job
}
with st.expander('DataFrame of Your Features'):
  pd.DataFrame(data, index=[0])
