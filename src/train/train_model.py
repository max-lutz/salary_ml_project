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
from sklearn.preprocessing import OneHotEncoder


from sklearn.model_selection import KFold, train_test_split, cross_validate

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, Trials
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
    },

    {
        'type': 'HuberRegressor',
        'epsilon': hp.uniform('HuberRegressor_epsilon', 1, 5),
        'max_iter': hp.quniform('HuberRegressor_max_iter', 1000, 20_000, 1000),
        'alpha': hp.loguniform('HuberRegressor_alpha', -20, 0),
    },

    {
        'type': 'PoissonRegressor',
        'alpha': hp.quniform('PoissonRegressor_alpha', 0, 5, 0.01),
        'max_iter': hp.quniform('PoissonRegressor_max_iter', 1000, 20_000, 1000),
    },

    {
        'type': 'GammaRegressor',
        'alpha': hp.quniform('GammaRegressor_alpha', 0, 5, 0.01),
        'max_iter': hp.quniform('GammaRegressor_max_iter', 1000, 20_000, 1000),
    },

    {
        'type': 'SVR',
        'kernel': hp.choice('SVR_kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']),
        'C': hp.quniform('SVR_C', 0, 5, 0.01),
        'epsilon': hp.quniform('SVR_epsilon', 0, 5, 0.01),
    },

    {
        'type': 'LGBMRegressor',
        'learning_rate':    hp.uniform('LGBMRegressor_learning_rate', 0.1, 1),
        'max_depth':        hp.quniform('LGBMRegressor_max_depth', 2, 100, 1),
        'min_child_weight': hp.quniform('LGBMRegressor_min_child_weight', 1, 50, 1),
        'colsample_bytree': hp.uniform('LGBMRegressor_colsample_bytree', 0.4, 1),
        'subsample':        hp.uniform('LGBMRegressor_subsample', 0.6, 1),
        'num_leaves':       hp.quniform('LGBMRegressor_num_leaves', 1, 200, 1),
        'min_split_gain':   hp.uniform('LGBMRegressor_min_split_gain', 0, 1),
        'reg_alpha':        hp.uniform('LGBMRegressor_reg_alpha', 0, 1),
        'reg_lambda':       hp.uniform('LGBMRegressor_reg_lambda', 0, 1),
        'n_estimators':     hp.quniform('LGBMRegressor_n_estimators', 10, 500, 10),
    },

    {
        'type': 'ElasticNet',
        'alpha': hp.quniform('ElasticNet_alpha', 0, 5, 0.01),
        'l1_ratio': hp.quniform('ElasticNet_l1_ratio', 0, 1, 0.01),
        'max_iter': hp.quniform('ElasticNet_max_iter', 1000, 20_000, 1000),
    },

    {
        'type': 'BayesianRidge',
        'max_iter': hp.quniform('BayesianRidgemax_iter', 1000, 20_000, 1000),
        'alpha_1': hp.loguniform('BayesianRidgea_lpha_1', -20, 0),
        'alpha_2': hp.loguniform('BayesianRidgea_lpha_2', -20, 0),
        'lambda_1': hp.loguniform('BayesianRidge_lambda_1', -20, 0),
        'lambda_2': hp.loguniform('BayesianRidge_lambda_2', -20, 0),
    },

    {
        'type': 'XGBRegressor',
        'max_depth': hp.quniform('XGBRegressor_max_depth', 3, 15, 1),
        'n_estimators': hp.quniform('XGBRegressor_n_estimators', 50, 300, 10),
        'colsample_bytree': hp.quniform('XGBRegressor_colsample_bytree', 0.5, 1.0, 0.1),
        'min_child_weight': hp.quniform('XGBRegressor_min_child_weight', 0, 10, 1),
        'subsample': hp.quniform('XGBRegressor_subsample', 0.5, 1.0, 0.1),
        'learning_rate': hp.quniform('XGBRegressor_learning_rate', 0.1, 0.3, 0.1),
        'gamma': hp.quniform('XGBRegressor_gamma', 0, 10, 0.1),
        'reg_alpha': hp.quniform('XGBRegressor_reg_alpha', 0, 10, 0.1),
        'reg_lambda': hp.quniform('XGBRegressor_reg_lambda', 0, 20, 0.1),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    },

    {
        'type': 'KernelRidge',
        'alpha': hp.quniform('KernelRidge_alpha', 0, 5, 0.01),
    },

]


def objective(params):
    scoring = {'neg_mean_squared_error': 'neg_mean_squared_error'}
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    regressor = params['type']

    del params['type']
    if regressor == 'Ridge':
        clf = Ridge(**params)
    elif regressor == 'HuberRegressor':
        clf = HuberRegressor(**params)
    elif regressor == 'PoissonRegressor':
        clf = PoissonRegressor(**params)
    elif regressor == 'GammaRegressor':
        clf = GammaRegressor(**params)
    elif regressor == 'SVR':
        clf = SVR(**params)
    elif regressor == 'LGBMRegressor':
        clf = LGBMRegressor(**params, verbosity=-1, force_row_wise=True)
    elif regressor == 'ElasticNet':
        clf = ElasticNet(**params)
    elif regressor == 'BayesianRidge':
        clf = BayesianRidge(**params)
    elif regressor == 'XGBRegressor':
        clf = XGBRegressor(**params)
    elif regressor == 'KernelRidge':
        clf = KernelRidge(**params)
    else:
        return 0

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.set_tag('model', regressor)
        mlflow.log_params(params)

        cv_score = cross_validate(clf, X, y, cv=folds, scoring=scoring, verbose=0, error_score="raise")
        rmse = round(np.sqrt(-cv_score['test_neg_mean_squared_error']).mean(), 6)
        mlflow.log_metric('RMSE', rmse)

    return {'loss': rmse, 'status': STATUS_OK}


if __name__ == "__main__":
    df = pd.read_csv('../../data/linkedin_jobs.csv')
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
        'HuberRegressor': HuberRegressor(max_iter=1000),
        'PoissonRegressor': PoissonRegressor(max_iter=10_000),
        'GammaRegressor': GammaRegressor(max_iter=1000),
        'SVR': SVR(),
        'LGBMRegressor': LGBMRegressor(verbosity=-1, force_row_wise=True),
        'ElasticNet': ElasticNet(),
        'BayesianRidge': BayesianRidge(),
        'XGBRegressor': XGBRegressor(),
        'KernelRidge': KernelRidge()
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
    models_preselected = df_model_perf[(df_model_perf['Mean error'] < df_model_perf['Mean error'].min()*1.2) &
                                       (df_model_perf['Median fit time'] < 1)].Model.to_list()

    regressor_search_space = [h for h in hyperparameters if h['type'] in models_preselected]
    search_space = hp.choice('regressor', regressor_search_space)

    trials = Trials()
    algo = tpe.suggest

    X = X_train_preprocessed
    y = y_train

    best_result = fmin(fn=objective, space=search_space, algo=algo, max_evals=1000, trials=trials)
