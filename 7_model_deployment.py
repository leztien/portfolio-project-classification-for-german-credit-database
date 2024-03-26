#!/usr/bin/env python


"""
main.py
$ pip install "fastapi[all]"
$ uvicorn main:app --reload
http://127.0.0.1:8000/redoc

POST requests in the bash console:

curl \
  --header "Content-Type: application/json" \
  --request POST \
  --data '{"values": ["A11", 6, "A34", "A43", 1169, "A65", "A75", 4, "A93", "A101", 4, "A121", 67, "A143", "A152", 2, "A173", 1, "A192", "A201"]}' \
  http://localhost:8000/predict
  
  
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: aplication/json' \
  -H 'Content-Type: application/json' \
  -d '{"values": ["A11", 6, "A34", "A43", 1169, "A65", "A75", 4, "A93", "A101", 4, "A121", 67, "A143", "A152", 2, "A173", 1, "A192", "A201"]}'
"""


import uvicorn  # not needed if  $ uvicorn main:app
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from typing import List, Any
from pydantic import BaseModel, validator
import joblib
from src.helper_utilities import load_data
import requests


app = FastAPI()
model = joblib.load('models/best_model.pkl')


def predict(data, model=model, threshold=0.37373737373737376):
    """Note that only the 0'th element gets to be returned"""
    return int((model.predict_proba([list(data)])[:, -1] >= threshold).astype(int)[0])


class Data(BaseModel):
    values: List[Any]

    @validator('values')
    def check_len(cls, v):
        if len(v) != 20:
            raise ValueError("len must be exactly 20")
        return v
    
    @validator('values')
    def check_types(cls, v):
        types = (str, int, str, str, int, str, str, int, str, str, int, str, int, str, str, int, str, int, str, str)
        assert all(isinstance(e, t) for e,t in zip(v, types)), "bad type"
        return v


@app.post("/predict")
async def process_data(data: Data):
    return {"prediction": predict(data.values)}


@app.get("/")
async def root():
    return HTMLResponse("Hello World")


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# or bash: $ uvicorn main:app --reload



# POST request: run this in  a different cell while the server is on
# Load the model and data
X = load_data(mode='predict', format='ndarray', introduce_nans=False, random_state=None)
x = list(X[0])

url = "http://localhost:8000/predict"  # or ...
url = 'http://127.0.0.1:8000/predict'

data = {"values": x}

response = requests.post(url, json=data)
print(response.json())

