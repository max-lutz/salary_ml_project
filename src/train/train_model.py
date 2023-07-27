'''


Usage: python train_model.py --in ../../data/linkedin_jobs.csv --n_eval 1


Setup rest api: 
mlflow models serve -m file:///home/maxou1909/Desktop/STREAMLIT_APPS/salary_ml_project/mlruns/dfac3ee2f7454f45b74b57ac25ae0c86/artifacts/salary_models -h 127.0.0.1 -p 8001 --env-manager=local

Query from rest api
curl -X POST -H "Content-Type:application/json" --data '{"dataframe_split":{"columns":["title", "location", "experience", "description"],"data":[["data analyst","chicago","ENTRY_LEVEL", "test big salary"]]}}' http://127.0.0.1:8001/invocations

'''

import sys
import argparse
import os

# import dependencies
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error as mse

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import mlflow


sys.path.append('../')  # nopep8
from utils.train_utils import *  # nopep8


# setup mlflow directory and experiment
mlflow_directory = f'file://{os.path.abspath("../../mlruns")}'

mlflow.set_tracking_uri(uri=mlflow_directory)
exp = mlflow.get_experiment_by_name(name='Linkedin_salary')
if not exp:
    experiment_id = mlflow.create_experiment(name='Linkedin_salary', artifact_location=mlflow_directory)
else:
    experiment_id = exp.experiment_id


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Query data from linkedin job posts")
    parser.add_argument("--in", dest="input", required=True,
                        metavar="input", type=str, help="Path of input file, example: ../../linkedin_jobs.csv")
    parser.add_argument('--n_evals', dest='n_evals', type=int, help='Number of models to test', default=10)

    print("Parsing arguments")
    args = parser.parse_args()
    return args.input, args.n_evals


def objective(params):
    _, scoring, folds = get_parameters_cross_validation()
    regressor = params['type']
    max_features = params['vectorizer_max_features']

    preprocessing = generate_preprocessing_pipeline(max_features=max_features)

    del params['type']
    del params['vectorizer_max_features']
    model = get_model_from_str(regressor, params)

    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('to_dense', DenseTransformer()),
        ('model', model)
    ])

    params['vectorizer_max_features'] = max_features
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.set_tag('model', regressor)
        mlflow.log_params(params)

        cv_score = cross_validate(pipeline, X_train, y_train, cv=folds, scoring=scoring,
                                  verbose=0, error_score="raise", return_estimator=True)
        rmse_train = round(np.sqrt(-cv_score['test_neg_mean_squared_error']).mean(), 6)
        rmse_val = round(np.sqrt(mse(y_true=y_val, y_pred=cv_score['estimator'][0].predict(X_val))), 6)
        rmse_test = round(np.sqrt(mse(y_true=y_test, y_pred=cv_score['estimator'][0].predict(X_test))), 6)

        mlflow.sklearn.log_model(sk_model=cv_score['estimator'][0],
                                 artifact_path='salary_models', registered_model_name=regressor)
        mlflow.log_metric('RMSE_train', rmse_train)
        mlflow.log_metric('RMSE_val', rmse_val)
        mlflow.log_metric('RMSE_test', rmse_test)

    return {'loss': rmse_val, 'status': STATUS_OK}


if __name__ == "__main__":
    skip = False
    input_path, n_evals = parse_arguments()
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(input_path)
    except:
        skip = True
        print(f"Cannot open file {input_path}")

    if (not skip):
        models_preselected = select_base_models(X_train, y_train)
        regressor_search_space = [h for h in get_hyperparameters_list() if h['type'] in models_preselected]
        search_space = hp.choice('regressor', regressor_search_space)
        trials = Trials()
        algo = tpe.suggest
        best_result = fmin(fn=objective, space=search_space, algo=algo, max_evals=n_evals, trials=trials)
