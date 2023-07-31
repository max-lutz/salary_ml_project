# Salary machine learning project

This project presents an end-to-end machine learning solution to problem of predicting salary from job offers posted on linkedin.

This projects consists of the following steps:
- Problem definition
- Data collection
- Data preprocessing and feature engineering
- Model selection, training, and fine-tuning
- Model evaluation
- Model deployment
- Model monitoring


How to use this reporitory:
1. Clone the reporitory
2. Run pip install -r requirements.txt
3. Train a new model: python src/train/train_model.py --in data/linkedin_jobs.csv --n_eval 25
4. Serve the model locally (find the mlflow model you want to serve): 
    mlflow models serve -m {path to mlflow model} -h 127.0.0.1 -p 8001 --env-manager=local
5. Monitor the model: python src/monitor/monitor_local_model.py --train_file data/train.zip --test_file data/test.zip
   

End to end pipeline (the full pipeline can take some time to run, especially the data generation part, depending on --n_queries):
1. Generate data: Run python src/data/generate_data.py --queries "data analyst, data scientist, data engineer" --out data/train.h5 --n_queries 10
2. Prepare the data: python src/features/build_dataset.py --in data/train.h5 --out data/train.zip
3. Train a new model: python src/train/train_model.py --in data/linkedin_jobs.csv --n_eval 25
4. Serve the model locally (find the mlflow model you want to serve): 
    mlflow models serve -m {path to mlflow model} -h 127.0.0.1 -p 8001 --env-manager=local
4. Monitor the model: python src/monitor/monitor_local_model.py --train_file data/train.zip --test_file data/test.zip
  
               