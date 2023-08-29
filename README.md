# Salary machine learning project

This project presents an end-to-end machine learning solution to problem of predicting salary from job offers posted on linkedin.

You can access the online app to monitor the model hosted on AWS: https://salary-ml-project.streamlit.app/

This projects consists of the following steps:
- **Problem definition**: Predicting the salary of job offers on linkedin
- **Data collection**: Web-scraping script on the linkedin website
- **Data preprocessing and feature engineering**: Extracting numbers from the job description to estimate the salary
- **Model selection, training, and fine-tuning**: Selecting base models with good performances and fast training out of 15 model classes. Training more than 100 models using bayesian optimization to fine-tune hyperparameters.
- **Model evaluation**: Mlflow for comparing model performances
- **Model deployment**: Local deployment using mlflow and online deployment on AWS using fastapi
- **Model monitoring**: Local monitoring on data quality, data drift and model performances. Online monitoring on API latency, and predictions errors.
<br/>

![salary_ml](https://github.com/max-lutz/salary_ml_project/assets/39080117/a1fc9d75-dbf1-4cc5-bb7b-6559e485ec38)

## How to use this repository:
1. Clone the repository
2. Run pip install -r requirements.txt
3. Train a new model: python src/train/train_model.py --in data/linkedin_jobs.csv --n_eval 25
4. Serve the model locally (find the mlflow model you want to serve): 
    mlflow models serve -m {path to mlflow model} -h 127.0.0.1 -p 8001 --env-manager=local
5. Monitor the model: python src/monitor/monitor_local_model.py --train_file data/train.zip --test_file data/test.zip
   

### Instructions for end-to-end pipeline 
The full pipeline can take some time to run, especially the data generation part, depending on --n_queries:
1. Generate data: Run python src/data/generate_data.py --queries "data analyst, data scientist, data engineer" --out data/train.h5 --n_queries 10
2. Prepare the data: python src/features/build_dataset.py --in data/train.h5 --out data/train.zip
3. Train a new model: python src/train/train_model.py --in data/linkedin_jobs.csv --n_eval 25
4. Serve the model locally (find the mlflow model you want to serve): 
    mlflow models serve -m {path to mlflow model} -h 127.0.0.1 -p 8001 --env-manager=local
4. Monitor the model: python src/monitor/monitor_local_model.py --train_file data/train.zip --test_file data/test.zip
  
               
