import streamlit as st
import pandas as pd
import pickle
import gzip
import requests
import numpy as np

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
  income = st.number_input('Annual Income', 100000, 10000000, 100000, 500000)
  loan_amount = st.number_input('Loan Amount', 300000, 50000000, 1000000, 100000)
  loan_term = st.slider('Loan Term', 2, 40, 25)
  score = st.number_input('Cibil Score', 200, 1000, 500, 50)
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
  loan = pd.DataFrame(data, index=[0])
  loan
data2 = {
  'no_of_dependents' : no_of_dep,
  'income_annum' : income,
  'loan_amount' : loan_amount,
  'loan_term' : loan_term,
  'cibil_score' : score,
  'residential_assets_value' : res_ass_val,
  'commercial_assets_value' : com_ass_val,
  'luxury_assets_value' : lux_ass_val,
  'bank_asset_value' : bank_ass_val,
}

loan_num = pd.DataFrame(data2, index=[0])

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('dummy_cols.pkl', 'rb') as f:
    dummy_columns = pickle.load(f)

cat = {
  'Education' : edu_level,
  'Self_Employed?' : job
}

loan_cat = pd.DataFrame(cat, index=[0])
loan_dummies = pd.get_dummies(loan_cat)
loan_dummies = loan_dummies.reindex(columns=dummy_columns, fill_value=0)

loan_num['asset_value_comb'] = loan_num['bank_asset_value'] + loan_num['commercial_assets_value'] + loan_num['luxury_assets_value'] + loan_num['residential_assets_value']
loan_num['loan_amount_per_income'] = loan_num['loan_amount'] / loan_num['income_annum']
loan_num['no_of_dependents'] = loan_num['no_of_dependents'] + 1
loan_num['income_annum_per_dependents'] = loan_num['income_annum'] / loan_num['no_of_dependents']


# loan_num = np.array(loan_num).reshape(1, -1)
scaled_loan = scaler.transform(loan_num)
scaled_df = pd.DataFrame(scaled_loan, columns=loan_num.columns, index=[0])

X = pd.concat([scaled_df, loan_dummies], axis=1)

with gzip.open('model.pkl.gz', 'rb') as f:
    model = pickle.load(f)
if st.button("Predict"):
  y = model.predict(X)
  if y == 'Approved':
    print('🎉 Congratulations, you are Eligible to get a Loan')
  else:
    print('😓 Sorry! you are not Eligbile for a Loan')
    
  
