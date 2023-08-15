'''
Script to download recent job offers from linkedin based on keyword given by the user

Usage: 
python monitor_local_model.py --test_file ../../data/test.zip
'''

import os
import shutil
import sys
import json
import requests
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData
from linkedin_jobs_scraper.filters import ExperienceLevelFilters

import datetime
from evidently import ColumnMapping, metrics
from evidently.metrics import ColumnDriftMetric, ColumnSummaryMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently.metrics import RegressionQualityMetric, RegressionPredictedVsActualPlot, RegressionErrorPlot
from evidently.metrics import DataDriftTable
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.metric_preset import TextOverviewPreset
from evidently.test_suite import TestSuite
from evidently.ui.dashboards import CounterAgg, DashboardPanelCounter, DashboardPanelPlot, PanelValue, PlotType, ReportFilter
from evidently.ui.workspace import Workspace, WorkspaceBase


from evidently.tests import TestColumnDrift, TestValueRange, TestValueRMSE, TestNumberOfOutRangeValues, TestShareOfOutRangeValues

from evidently.tests import TestHighlyCorrelatedColumns, TestTargetFeaturesCorrelations, TestPredictionFeaturesCorrelations
from evidently.tests import TestCorrelationChanges, TestNumberOfDriftedColumns, TestShareOfDriftedColumns

from evidently.descriptors import TextLength, TriggerWordsPresence, OOV, NonLetterCharacterPercentage, SentenceCount, WordCount, Sentiment, RegExp


sys.path.append('../src/')  # nopep8
from utils.preprocess_utils import generate_salary, preprocess_text, clean_dataset  # nopep8


def predict(data):
    url = 'http://0.0.0.0:9000/'
    data_dict = {"title": data[0], "location": data[1], "experience": data[2], "description": data[3]}
    data_dict = {"data": json.dumps(data_dict)}

    print(data_dict)

    headers = {
        'accept': 'application/json',
        'content-type': 'application/x-www-form-urlencoded',
    }

    data_json = json.dumps(data_dict)
    print(data_json)
    response = requests.post('http://0.0.0.0:9000/', params=data_dict, headers=headers)
    print(response)
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
            * The code can be accessed at [code](https://github.com/max-lutz/ML-regression-tool).
            * Click on how to use this app to get more explanation.
            """)

title_spacer2, title_2, title_spacer_2 = st.columns((.1, ROW, .1))
with title_2:
    with st.expander("How to use this app"):
        st.markdown("""
            """)
        st.write("")

st.sidebar.header('Test API')
title = st.sidebar.text_input('Job title')
location = st.sidebar.selectbox('Location', ['Chicago'])
experience = st.sidebar.selectbox(
    'Experience level', ['ENTRY_LEVEL', 'ASSOCIATE', 'MID_SENIOR', 'DIRECTOR', 'EXECUTIVE'])
description = st.sidebar.text_input('Job description')

data = [title, location, experience, description]
st.sidebar.write(predict(data))
