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
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData
from linkedin_jobs_scraper.filters import ExperienceLevelFilters

import datetime
from evidently import ColumnMapping, metrics
from evidently.metrics import ColumnDriftMetric, ColumnSummaryMetric, DatasetDriftMetric, DatasetMissingValuesMetric, DataDriftTable
from evidently.metrics import RegressionQualityMetric
from evidently.metrics import RegressionPredictedVsActualScatter
from evidently.metrics import RegressionPredictedVsActualPlot
from evidently.metrics import RegressionErrorPlot
from evidently.metrics import RegressionAbsPercentageErrorPlot
from evidently.metrics import RegressionErrorDistribution
from evidently.metrics import RegressionErrorNormality
from evidently.metrics import RegressionTopErrorMetric
from evidently.metrics import RegressionErrorBiasTable
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.ui.dashboards import CounterAgg, DashboardPanelCounter, DashboardPanelPlot, PanelValue, PlotType, ReportFilter
from evidently.ui.workspace import Workspace, WorkspaceBase

sys.path.append('../')  # nopep8
from utils.data_utils import on_error, on_end, run_queries, get_scrapper  # nopep8
from utils.preprocess_utils import generate_salary, preprocess_text, clean_dataset  # nopep8


search_keyword = ''
EXP_LVL_STR = ['ENTRY_LEVEL', 'ASSOCIATE', 'MID_SENIOR', 'DIRECTOR', 'EXECUTIVE']
EXP_LVL_CLASS = [ExperienceLevelFilters.ENTRY_LEVEL, ExperienceLevelFilters.ASSOCIATE,
                 ExperienceLevelFilters.MID_SENIOR, ExperienceLevelFilters.DIRECTOR, ExperienceLevelFilters.EXECUTIVE]
EXP_LVL_INDEX = 0


WORKSPACE = "../../workspace"
YOUR_PROJECT_NAME = "Linkedin jobs salary"
YOUR_PROJECT_DESCRIPTION = "Monitor salary prediction on the linkedin jobs dataset"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Query data from linkedin job posts")
    parser.add_argument("--train_file", dest="train_file", type=str, required=True,
                        help="Path of train file to monitor, example: ../../train.zip")
    parser.add_argument("--test_file", dest="test_file", type=str, required=True,
                        help="Path of test file to monitor, example: ../../test.zip")

    print("Parsing arguments")
    args = parser.parse_args()
    return args.train_file, args.test_file


def predict(data):
    url = 'http://127.0.0.1:8001/invocations'
    data_dict = {
        "dataframe_split":
            {
                "columns": ["title", "location", "experience", "description"],
                "data": [data]
            }
    }

    data_json = json.dumps(data_dict)
    response = requests.post(url, data=data_json, headers={"Content-Type": "application/json"})
    return response.json()


def on_data(data: EventData):
    row = [[search_keyword, data.title, data.company, data.link, data.place, data.description,
           data.date, EXP_LVL_STR[EXP_LVL_INDEX]]]
    df = pd.DataFrame(row, columns=['search_keyword', 'title', 'company', 'link', 'location',
                      'description', 'date', 'experience'])
    df = generate_salary(df)
    df = preprocess_text(df)
    df = clean_dataset(df, dropna=False)

    data = df.iloc[0].to_list()

    predicted_salary = int(predict(data[0:-1])['predictions'][0]/1000)*1000
    if (data[-1]):
        print('Salary exists')
        print(f"Title {data[0]}, Predicted salary {predicted_salary}, True salary {data[-1]}")
    else:
        print(f"Title {data[0]}, Predicted salary {predicted_salary}, True salary {data[-1]}")


# https://github.com/evidentlyai/evidently/blob/dd6ccb8989e38ceff1b9c27b41d4ae2b10a5fef6/src/evidently/ui/demo_project.py#L75


def create_report(train, test, i: int):
    target = "target"
    prediction = "prediction"

    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = prediction

    data_drift_report = Report(
        metrics=[
            DatasetDriftMetric(columns=['target', 'prediction']),
            DatasetMissingValuesMetric(),
            DataDriftTable(columns=['target', 'prediction']),
            ColumnDriftMetric(column_name="target", stattest="wasserstein"),
            ColumnSummaryMetric(column_name="target"),
            ColumnDriftMetric(column_name="prediction", stattest="wasserstein"),
            ColumnSummaryMetric(column_name="prediction"),
            RegressionQualityMetric(),
            RegressionPredictedVsActualPlot(),
            RegressionErrorPlot()
        ],
        timestamp=datetime.datetime.now() + datetime.timedelta(days=i),
    )

    data_drift_report.run(reference_data=train,
                          current_data=test.iloc[100 * i: 100 * (i + 1), :],
                          column_mapping=column_mapping)
    return data_drift_report


def create_test_suite(train, test, i: int):
    data_drift_test_suite = TestSuite(
        tests=[DataDriftTestPreset()],
        timestamp=datetime.datetime.now() + datetime.timedelta(days=i),
    )

    data_drift_test_suite.run(reference_data=train, current_data=test.iloc[100 * i: 100 * (i + 1), :])
    return data_drift_test_suite


def create_project(workspace: WorkspaceBase):
    project = workspace.create_project(YOUR_PROJECT_NAME)
    project.description = YOUR_PROJECT_DESCRIPTION
    project.dashboard.add_panel(
        DashboardPanelCounter(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            agg=CounterAgg.NONE,
            title="Linkedin job offers salary predictions",
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Model Calls",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetMissingValuesMetric",
                field_path=DatasetMissingValuesMetric.fields.current.number_of_rows,
                legend="count",
            ),
            text="count",
            agg=CounterAgg.SUM,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Share of Drifted Features",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetDriftMetric",
                field_path="share_of_drifted_columns",
                legend="share",
            ),
            text="share",
            agg=CounterAgg.LAST,
            size=1,
        )
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Target and Prediction",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ColumnSummaryMetric",
                    field_path="current_characteristics.mean",
                    metric_args={"column_name.name": "target"},
                    legend="Target (daily mean)",
                ),
                PanelValue(
                    metric_id="ColumnSummaryMetric",
                    field_path="current_characteristics.mean",
                    metric_args={"column_name.name": "prediction"},
                    legend="Prediction (daily mean)",
                ),
            ],
            plot_type=PlotType.LINE,
            size=2,
        )
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="MAE",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="RegressionQualityMetric",
                    field_path=metrics.RegressionQualityMetric.fields.current.mean_abs_error,
                    legend="MAE",
                ),
            ],
            plot_type=PlotType.LINE,
            size=1,
        )
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="MAPE",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="RegressionQualityMetric",
                    field_path=metrics.RegressionQualityMetric.fields.current.mean_abs_perc_error,
                    legend="MAPE",
                ),
            ],
            plot_type=PlotType.LINE,
            size=1,
        )
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Target: Wasserstein drift distance",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "target"},
                    field_path=ColumnDriftMetric.fields.drift_score,
                    legend="Drift Score",
                ),
            ],
            plot_type=PlotType.BAR,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Predictions: Wasserstein drift distance",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "prediction"},
                    field_path=ColumnDriftMetric.fields.drift_score,
                    legend="Drift Score",
                ),
            ],
            plot_type=PlotType.BAR,
            size=1,
        )
    )
    project.save()
    return project


def create_demo_project(train, test, workspace: str):
    ws = Workspace.create(workspace)
    project = create_project(ws)

    for i in range(0, 5):
        report = create_report(train, test, i=i)
        ws.add_report(project.id, report)

        test_suite = create_test_suite(train, test, i=i)
        ws.add_test_suite(project.id, test_suite)


if __name__ == '__main__':

    train_file, test_file = parse_arguments()
    if ((train_file is not None) and (test_file is not None)):
        df_train = pd.read_csv(train_file).iloc[0:500]
        df_test = pd.read_csv(test_file).iloc[500:1000]

        print("Preparing data")
        for dataset in [df_train, df_test]:
            predictions = []
            for i in tqdm(range(len(dataset))):
                predictions.append(int(predict(dataset.iloc[i, 1:-1].values.tolist())['predictions'][0]/1000)*1000)
            dataset['prediction'] = predictions

        if (os.path.exists("workspace")):
            print("Folder exixts")
            shutil.rmtree("workspace")
        create_demo_project(df_train, df_test, "workspace")

        subprocess.run(["evidently", "ui"])

    else:
        print("Start scrapping to generate data...")
        # scraper = get_scrapper(slow=5)

        # # Add event listeners
        # scraper.on(Events.DATA, on_data)
        # scraper.on(Events.ERROR, on_error)
        # scraper.on(Events.END, on_end)

        # run_queries(scraper, ['data scientist', 'data engineer', 'data analyst'], ["Chicago"], 10, debug=False)
