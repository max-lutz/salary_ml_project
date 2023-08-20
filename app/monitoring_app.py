'''
Script to download recent job offers from linkedin based on keyword given by the user

Usage: 
streamlit run monitoring_app.py
'''


import json
import requests
import pandas as pd
from tqdm import tqdm

import streamlit as st
import time
import plotly.express as px


def predict(data):
    url = 'https://iovdak7a527hkhw3lm2b7ao2r40cspgo.lambda-url.us-east-1.on.aws/'
    data_dict = {"title": data[0], "location": data[1], "experience": data[2], "description": data[3]}
    data_dict = {"data": json.dumps(data_dict)}

    headers = {
        'accept': 'application/json',
        'content-type': 'application/x-www-form-urlencoded',
    }

    data_json = json.dumps(data_dict)
    response = requests.post(url, params=data_dict, headers=headers)
    return response.json()


# configuration of the page
st.set_page_config(layout="wide")


SPACER = .2
ROW = 1

title_spacer1, title, title_spacer_2 = st.columns((.1, ROW, .1))
with title:
    st.title('Salary model monitoring tool')
    st.markdown("""
            This app allows you monitor the salary model hosted on AWS
            * The code can be accessed at [code](https://github.com/max-lutz/salary_ml_project).
            * Click on how to use this app to get more explanation.
            """)

title_spacer2, title_2, title_spacer_2 = st.columns((.1, ROW, .1))
with title_2:
    with st.expander("How to use this app"):
        st.markdown("""
            """)
        st.write("")


st.sidebar.header('Test AWS API')
run_api_test = False
run_api_test = st.sidebar.button('Run aws api test')
if (run_api_test):
    df_test = pd.read_csv("data/test.zip").sample(3)
    api_latency = []

    for dataset in [df_test]:
        predictions = []
        for i in tqdm(range(len(dataset))):
            start_time = time.time()
            predictions.append(int(predict(dataset.iloc[i, 1:-1].values.tolist())['predictions'][0]/1000)*1000)
            api_latency.append(time.time()-start_time)
        dataset['prediction'] = predictions
        dataset['latency'] = api_latency
        dataset['rmse'] = abs(dataset['prediction'] - dataset['target'])
        dataset['id_plot'] = [int(i) for i in range(len(dataset))]

    st.header("Monitoring dashboard")
    st.subheader("Latency")
    col1, col2, col3 = st.columns(3)
    with (col1):
        st.markdown(f"##### Request sent: {len(df_test)}")
    with (col2):
        st.markdown(f"##### Average latency: {round(df_test['latency'].mean(), 2)} seconds")
    with (col3):
        st.markdown(f"##### Latency [99 percentile] sent: {round(df_test['latency'].quantile(0.99), 2)} seconds")

    fig = px.line(df_test, x="id_plot", y="latency", title='Latency in the last predictions')
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.subheader("Predictions errors")
    col1_1, col2_1, col3_1 = st.columns(3)
    with (col1_1):
        st.markdown(f"##### Average salary predicted: ${round(df_test['prediction'].mean(), 0)}")
    with (col2_1):
        st.markdown(f"##### Average RMSE: {round(df_test['rmse'].mean(), 2)}")
    with (col3_1):
        st.markdown(f"##### RMSE [99 percentile]: {round(df_test['rmse'].quantile(0.99), 0)}")

    df = pd.melt(df_test, id_vars=['id_plot'], value_vars=['prediction', 'target'])
    fig = px.line(df, x="id_plot", y="value", color="variable", symbol="variable", title='True value vs prediction')
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    fig = px.line(df_test, x="id_plot", y="rmse", title='RMSE')
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.subheader('Last predictions made')
    st.write(df_test)


st.sidebar.header('User input')
title = st.sidebar.text_input('Job title')
location = st.sidebar.selectbox('Location', ['Chicago'])
experience = st.sidebar.selectbox(
    'Experience level', ['ENTRY_LEVEL', 'ASSOCIATE', 'MID_SENIOR', 'DIRECTOR', 'EXECUTIVE'])
description = st.sidebar.text_input('Job description')
data = [title, location, experience, description]

make_prediction = False
make_prediction = st.sidebar.button('Make prediction')
if (make_prediction):
    st.sidebar.write(f"Salary predicted: ${int(predict(data)['predictions'][0]/1000)*1000}")
    make_prediction = False
