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

