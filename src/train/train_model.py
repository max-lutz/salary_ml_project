import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# import dependencies
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, Ridge, SGDRegressor, HuberRegressor, PoissonRegressor, GammaRegressor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder


from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.metrics import mean_squared_error

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, Trials
import mlflow


# setup mlflow directory and experiment
mlflow_directory = f'file://{os.path.abspath("../mlruns")}'

mlflow.set_tracking_uri(uri=mlflow_directory)
exp = mlflow.get_experiment_by_name(name='Linkedin_salary')
if not exp:
    experiment_id = mlflow.create_experiment(name='Linkedin_salary', artifact_location=mlflow_directory)
else:
    experiment_id = exp.experiment_id


if __name__ == "__main__":
    df = pd.read_csv('../data/linkedin_jobs.csv')
    train, test = train_test_split(df, test_size=0.3, stratify=df['experience'], random_state=0)
    test, val = train_test_split(test, test_size=0.5, stratify=test['experience'], random_state=0)

    X_train, y_train = train.drop(columns=['id', 'target']), train['target'].to_numpy()
    X_val, y_val = val.drop(columns=['id', 'target']), val['target'].to_numpy()
    X_test, y_test = test.drop(columns=['id', 'target']), test['target'].to_numpy()

    text_cols = ['title', 'description']
    cat_cols = ['location', 'experience']

    # combine text columns in one new column because TfidfVectorizer does not accept multiple columns
    if (len(text_cols) != 0):
        X_train['text'] = X_train[text_cols].astype(str).agg(' '.join, axis=1)
        X_val['text'] = X_val[text_cols].astype(str).agg(' '.join, axis=1)
        X_test['text'] = X_test[text_cols].astype(str).agg(' '.join, axis=1)
        text_cols = "text"

    preprocessing = make_column_transformer(
        (OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False), cat_cols),
        (TfidfVectorizer(strip_accents='ascii', max_features=1000), text_cols)
    )
    # use max_features in vectorizer gridsearchCV

    preprocessing.fit(X_train)
    X_train_preprocessed = preprocessing.fit_transform(X_train).toarray()
    X_val_preprocessed = preprocessing.transform(X_val).toarray()

    models_list = {
        'Linear regressor': LinearRegression(),
        'Ridge': Ridge(),
        'SGDRegressor': SGDRegressor(),
        'HuberRegressor': HuberRegressor(),
        'QuantileRegressor': QuantileRegressor(),
        'PoissonRegressor': PoissonRegressor(),
        'GammaRegressor': GammaRegressor(),
        'Random Forest': RandomForestRegressor(),
        'SVR': SVR(),
        'LGBMRegressor': LGBMRegressor(verbosity=-1, force_row_wise=True),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'BayesianRidge': BayesianRidge(),
        'GradientBoostingRegressor': GradientBoostingRegressor()
    }

    scoring = {'max_error': 'max_error', 'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
    columns = ['Model', 'Median fit time', 'Mean max error',
               'Std max error', 'Mean error', 'Std mean error', 'Mean r2', 'Std r2']

    folds = KFold(n_splits=5, shuffle=True, random_state=0)

    model_perf_matrix = []
    predictions = pd.DataFrame()
    for model_name, model in tqdm(models_list.items()):
        pipeline = Pipeline([
            ('model', model)
        ])

        cv_score = cross_validate(pipeline, X_train_preprocessed, y_train, cv=folds,
                                  scoring=scoring, verbose=0, error_score="raise")
        # cv_score = np.sqrt(-cross_val_score(pipeline, X, y, cv=folds, scoring=scoring));
        model_perf_matrix.append([model_name, round(cv_score['fit_time'].mean(), 3),
                                  round(cv_score['test_max_error'].mean(), 4), round(
                                      cv_score['test_max_error'].std(), 4),
                                  round(np.sqrt(-cv_score['test_neg_mean_squared_error']).mean(),
                                        4), round(np.sqrt(-cv_score['test_neg_mean_squared_error']).std(), 4),
                                  round(cv_score['test_r2'].mean(), 4), round(cv_score['test_r2'].std(), 4)])

        pipeline.fit(X_train_preprocessed, y_train)
        predictions[model_name] = pipeline.predict(X_val_preprocessed).T

    df_model_perf = pd.DataFrame(model_perf_matrix, columns=columns)
    # df_model_perf
