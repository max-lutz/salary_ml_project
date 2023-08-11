from mangum import Mangum
import uvicorn
from fastapi.responses import JSONResponse
from fastapi import FastAPI
import os
import json
import pickle
import requests
import pandas as pd

import nltk
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')


app = FastAPI()
handler = Mangum(app)

# Load the saved model
pipeline_path = "models/model.pkl"
with open(pipeline_path, 'rb') as file:
    pipeline = pickle.load(file)

if not os.path.exists('nltkdata'):
    os.mkdir('nltkdata')
    nltk.download('wordnet', download_dir='nltkdata')
    nltk.download('stopwords', download_dir='nltkdata')
    nltk.download('omw-1.4', download_dir='nltkdata')

nltk.data.path.append("nltkdata")


def predict(data):

    data = json.loads(data)
    print(data)
    df = pd.DataFrame([[data['title'], data['location'], data['experience'], data['description']]],
                      columns=["title", "location", "experience", "description"],)
    prediction = pipeline.predict(df)
    return prediction


@app.get('/')
def my_function(data: str):
    pred = predict(data)
    return JSONResponse({"predictions": list(pred)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
