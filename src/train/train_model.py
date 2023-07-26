'''


Usage: python train_model.py --in ../../data/linkedin_jobs.csv --n_eval 1


Setup rest api: 
mlflow models serve -m file:///home/maxou1909/Desktop/STREAMLIT_APPS/salary_ml_project/mlruns/dfac3ee2f7454f45b74b57ac25ae0c86/artifacts/salary_models -h 127.0.0.1 -p 8001 --env-manager=local

Query from rest api
curl -X POST -H "Content-Type:application/json" --data '{"dataframe_split":{"columns":["title", "location", "experience", "description"],"data":[["data analyst","chicago","ENTRY_LEVEL", "test big salary"]]}}' http://127.0.0.1:8001/invocations

'''

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# import dependencies
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, BayesianRidge, Ridge, HuberRegressor, PoissonRegressor, GammaRegressor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.base import TransformerMixin

from sklearn.model_selection import KFold, train_test_split, cross_validate

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import mlflow


# setup mlflow directory and experiment
mlflow_directory = f'file://{os.path.abspath("../../mlruns")}'

mlflow.set_tracking_uri(uri=mlflow_directory)
exp = mlflow.get_experiment_by_name(name='Linkedin_salary')
if not exp:
    experiment_id = mlflow.create_experiment(name='Linkedin_salary', artifact_location=mlflow_directory)
else:
    experiment_id = exp.experiment_id

hyperparameters = [
    {
        'type': 'Ridge',
        'solver': hp.choice('ridge_solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
        'alpha': hp.quniform('ridge_alpha', 0, 5, 0.01),
        'vectorizer_max_features': scope.int(hp.quniform('Ridge_vectorizer_max_features', 100, 10_000, 100)),
    },

    {
        'type': 'HuberRegressor',
        'epsilon': hp.uniform('HuberRegressor_epsilon', 1, 5),
        'max_iter': scope.int(hp.quniform('HuberRegressor_max_iter', 1000, 20_000, 1000)),
        'alpha': hp.loguniform('HuberRegressor_alpha', -20, 0),
        'vectorizer_max_features': scope.int(hp.quniform('HuberRegressor_vectorizer_max_features', 100, 10_000, 100)),
    },

    {
        'type': 'PoissonRegressor',
        'alpha': hp.quniform('PoissonRegressor_alpha', 0, 5, 0.01),
        'max_iter': scope.int(hp.quniform('PoissonRegressor_max_iter', 1000, 20_000, 1000)),
        'vectorizer_max_features': scope.int(hp.quniform('PoissonRegressor_vectorizer_max_features', 100, 10_000, 100)),
    },

    {
        'type': 'GammaRegressor',
        'alpha': hp.quniform('GammaRegressor_alpha', 0, 5, 0.01),
        'max_iter': scope.int(hp.quniform('GammaRegressor_max_iter', 1000, 20_000, 1000)),
        'vectorizer_max_features': scope.int(hp.quniform('GammaRegressor_vectorizer_max_features', 100, 10_000, 100)),
    },

    {
        'type': 'SVR',
        'kernel': hp.choice('SVR_kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']),
        'C': hp.quniform('SVR_C', 0, 5, 0.01),
        'epsilon': hp.quniform('SVR_epsilon', 0, 5, 0.01),
        'vectorizer_max_features': scope.int(hp.quniform('SVR_vectorizer_max_features', 100, 10_000, 100)),
    },

    {
        'type': 'LGBMRegressor',
        'learning_rate':    hp.uniform('LGBMRegressor_learning_rate', 0.0001, 0.25),
        'max_depth':        scope.int(hp.quniform('LGBMRegressor_max_depth', 2, 200, 1)),
        'colsample_bytree': hp.uniform('LGBMRegressor_colsample_bytree', 0.4, 1),
        'subsample':        hp.uniform('LGBMRegressor_subsample', 0.6, 1),
        'num_leaves':       scope.int(hp.quniform('LGBMRegressor_num_leaves', 1, 200, 1)),
        'min_split_gain':   hp.uniform('LGBMRegressor_min_split_gain', 0, 1),
        'reg_alpha':        hp.uniform('LGBMRegressor_reg_alpha', 0, 1),
        'reg_lambda':       hp.uniform('LGBMRegressor_reg_lambda', 0, 1),
        'n_estimators':     scope.int(hp.quniform('LGBMRegressor_n_estimators', 10, 500, 10)),
        'vectorizer_max_features': scope.int(hp.quniform('LGBMRegressor_vectorizer_max_features', 100, 10_000, 100)),
    },

    {
        'type': 'ElasticNet',
        'alpha': hp.quniform('ElasticNet_alpha', 0, 5, 0.01),
        'l1_ratio': hp.quniform('ElasticNet_l1_ratio', 0, 1, 0.01),
        'max_iter': scope.int(hp.quniform('ElasticNet_max_iter', 1000, 20_000, 1000)),
        'vectorizer_max_features': scope.int(hp.quniform('ElasticNet_vectorizer_max_features', 100, 10_000, 100)),
    },

    {
        'type': 'BayesianRidge',
        'max_iter': scope.int(hp.quniform('BayesianRidgemax_iter', 1000, 20_000, 1000)),
        'alpha_1': hp.loguniform('BayesianRidgea_lpha_1', -20, 0),
        'alpha_2': hp.loguniform('BayesianRidgea_lpha_2', -20, 0),
        'lambda_1': hp.loguniform('BayesianRidge_lambda_1', -20, 0),
        'lambda_2': hp.loguniform('BayesianRidge_lambda_2', -20, 0),
        'vectorizer_max_features': scope.int(hp.quniform('BayesianRidge_vectorizer_max_features', 100, 10_000, 100)),
    },

    {
        'type': 'XGBRegressor',
        'max_depth': scope.int(hp.quniform('XGBRegressor_max_depth', 3, 15, 1)),
        'n_estimators': scope.int(hp.quniform('XGBRegressor_n_estimators', 10, 500, 10)),
        'colsample_bytree': hp.uniform('XGBRegressor_colsample_bytree', 0.5, 1.0),
        'min_child_weight': scope.int(hp.quniform('XGBRegressor_min_child_weight', 0, 10, 1)),
        'subsample': hp.uniform('XGBRegressor_subsample', 0.5, 1.0),
        'learning_rate': hp.uniform('XGBRegressor_learning_rate', 0.0001, 0.25),
        'gamma': hp.uniform('XGBRegressor_gamma', 0, 1),
        'reg_alpha': hp.quniform('XGBRegressor_reg_alpha', 0, 10, 0.1),
        'reg_lambda': hp.quniform('XGBRegressor_reg_lambda', 0, 20, 0.1),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'vectorizer_max_features': scope.int(hp.quniform('XGBRegressor_vectorizer_max_features', 100, 10_000, 100)),
    },

    {
        'type': 'KernelRidge',
        'alpha': hp.quniform('KernelRidge_alpha', 0, 5, 0.01),
        'vectorizer_max_features': scope.int(hp.quniform('KernelRidge_vectorizer_max_features', 100, 10_000, 100)),
    },

]


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray()


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
    scoring = {'neg_mean_squared_error': 'neg_mean_squared_error'}
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    regressor = params['type']
    max_features = params['vectorizer_max_features']

    text_col_1 = 'title'
    text_col_2 = 'description'
    cat_cols = ['location', 'experience']

    preprocessing = make_column_transformer(
        (OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False), cat_cols),
        (TfidfVectorizer(strip_accents='ascii', stop_words='english', max_features=max_features), text_col_1),
        (TfidfVectorizer(strip_accents='ascii', stop_words='english', max_features=max_features), text_col_2)
    )

    del params['type']
    del params['vectorizer_max_features']
    if regressor == 'Ridge':
        model = Ridge(**params)
    elif regressor == 'HuberRegressor':
        model = HuberRegressor(**params)
    elif regressor == 'PoissonRegressor':
        model = PoissonRegressor(**params)
    elif regressor == 'GammaRegressor':
        model = GammaRegressor(**params)
    elif regressor == 'SVR':
        model = SVR(**params)
    elif regressor == 'LGBMRegressor':
        model = LGBMRegressor(**params, verbosity=-1, force_row_wise=True)
    elif regressor == 'ElasticNet':
        model = ElasticNet(**params)
    elif regressor == 'BayesianRidge':
        model = BayesianRidge(**params)
    elif regressor == 'XGBRegressor':
        model = XGBRegressor(**params)
    elif regressor == 'KernelRidge':
        model = KernelRidge(**params)
    else:
        return 0

    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('to_dense', DenseTransformer()),
        ('model', model)
    ])

    params['vectorizer_max_features'] = max_features
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.set_tag('model', regressor)
        mlflow.log_params(params)

        cv_score = cross_validate(pipeline, X, y, cv=folds, scoring=scoring,
                                  verbose=0, error_score="raise", return_estimator=True)
        rmse = round(np.sqrt(-cv_score['test_neg_mean_squared_error']).mean(), 6)

        mlflow.sklearn.log_model(sk_model=cv_score['estimator'][0],
                                 artifact_path='salary_models', registered_model_name=regressor)
        mlflow.log_metric('RMSE', rmse)

    return {'loss': rmse, 'status': STATUS_OK}


def load_data(input_path):
    df = pd.read_csv(input_path)
    train, test = train_test_split(df, test_size=0.3, stratify=df['experience'], random_state=0)
    test, val = train_test_split(test, test_size=0.5, stratify=test['experience'], random_state=0)

    X_train, y_train = train.drop(columns=['id', 'target']), train['target'].to_numpy()
    X_val, y_val = val.drop(columns=['id', 'target']), val['target'].to_numpy()
    X_test, y_test = test.drop(columns=['id', 'target']), test['target'].to_numpy()
    return X_train, y_train, X_val, y_val, X_test, y_test


# def aggregate_text_columns(X_train, X_val, X_test, text_cols):
#     # combine text columns in one new column because TfidfVectorizer does not accept multiple columns
#     if (len(text_cols) != 0):
#         X_train['text'] = X_train[text_cols].astype(str).agg(' '.join, axis=1)
#         X_val['text'] = X_val[text_cols].astype(str).agg(' '.join, axis=1)
#         X_test['text'] = X_test[text_cols].astype(str).agg(' '.join, axis=1)
#         text_cols = "text"
#     return X_train, X_val, X_test, text_cols


if __name__ == "__main__":
    skip = False
    text_col_1 = 'title'
    text_col_2 = 'description'
    cat_cols = ['location', 'experience']
    input_path, n_evals = parse_arguments()
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(input_path)
    except:
        skip = True
        print(f"Cannot open file {input_path}")

    if (not skip):
        # X_train, X_val, X_test, text_cols = aggregate_text_columns(X_train, X_val, X_test, text_cols)

        preprocessing = make_column_transformer(
            (OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False), cat_cols),
            (TfidfVectorizer(strip_accents='ascii', stop_words='english', max_features=1000), text_col_1),
            (TfidfVectorizer(strip_accents='ascii', stop_words='english', max_features=1000), text_col_2)
        )
        # use max_features in vectorizer gridsearchCV

        # preprocessing.fit(X_train)
        # X_train_preprocessed = preprocessing.fit_transform(X_train).toarray()
        # X_val_preprocessed = preprocessing.transform(X_val).toarray()

        models_list = {
            'Linear regressor': LinearRegression(),
            'Ridge': Ridge(),
            # 'HuberRegressor': HuberRegressor(max_iter=1000),
            # 'PoissonRegressor': PoissonRegressor(max_iter=10_000),
            # 'GammaRegressor': GammaRegressor(max_iter=1000),
            # 'SVR': SVR(),
            # 'LGBMRegressor': LGBMRegressor(verbosity=-1, force_row_wise=True),
            # 'ElasticNet': ElasticNet(),
            # 'BayesianRidge': BayesianRidge(),
            # 'XGBRegressor': XGBRegressor(),
            # 'KernelRidge': KernelRidge()
        }
        scoring = {'max_error': 'max_error', 'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
        columns = ['Model', 'Median fit time', 'Mean error']
        folds = KFold(n_splits=5, shuffle=True, random_state=0)

        model_perf_matrix = []
        predictions = pd.DataFrame()
        for model_name, model in tqdm(models_list.items()):
            pipeline = Pipeline([
                ('preprocessing', preprocessing),
                ('to_dense', DenseTransformer()),
                ('model', model)
            ])

            cv_score = cross_validate(pipeline, X_train, y_train, cv=folds,
                                      scoring=scoring, verbose=0, error_score="raise")
            model_perf_matrix.append([model_name, round(cv_score['fit_time'].mean(), 3),
                                      round(np.sqrt(-cv_score['test_neg_mean_squared_error']).mean(), 4)])

            pipeline.fit(X_train, y_train)
            predictions[model_name] = pipeline.predict(X_val).T

        df_model_perf = pd.DataFrame(model_perf_matrix, columns=columns)
        models_preselected = df_model_perf[(df_model_perf['Mean error'] < df_model_perf['Mean error'].min()*1.2) &
                                           (df_model_perf['Median fit time'] < 1)].Model.to_list()

        # print(df_model_perf)

        regressor_search_space = [h for h in hyperparameters if h['type'] in models_preselected]
        search_space = hp.choice('regressor', regressor_search_space)

        trials = Trials()
        algo = tpe.suggest

        X = X_train
        y = y_train

        best_result = fmin(fn=objective, space=search_space, algo=algo, max_evals=n_evals, trials=trials)
