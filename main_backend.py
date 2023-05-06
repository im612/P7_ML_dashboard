# backend/main.py

# " this is where we put the FastAPI endpoints"
import os

from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn
# import gunicorn
# import numpy as np
from model import get_probability_df
from model import get_prediction
from model import get_threshold
from model import get_indnames, load_indnames

# asyncronous models
# https://asgi.readthedocs.io/en/latest/

# Da https://www.youtube.com/watch?v=h5wLuVDr0oc
# Da https://testdriven.io/blog/fastapi-streamlit/
# https://www.youtube.com/watch?v=IvHCxycjeR0 DF


app = FastAPI()

class Id(BaseModel):
    id: str

# class Id(BaseModel):
#     id: str

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/probability2")
def pred_prob(iddata: Id):
    proba = float(get_probability_df(int(iddata.id)))
    return {"probability": proba}

@app.post("/probability")
def pred_prob(iddata):
    proba = float(get_probability_df(int(iddata)))
    return {"probability": proba}

@app.post("/prediction")
def prediction(iddata: Id):
    pred = float(get_prediction(int(iddata.id)))
    return {"prediction": pred}

@app.post("/seuil")
def prediction(iddata: Id):
    val = float(get_threshold())
    return {"seuil": val}

@app.post("/indnames")
def ind_names():
    val = get_indnames()
    # val = load_indnames()
    return {"listindnames": val}

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8080)
#     uvicorn.run("main:app-1container-nonfunziona", host="backend", port=8080)
    # gunicorn.run("main:app-1container-nonfunziona", host="0.0.0.0", port=8080)

# print(os.system("""host "0.0.0.0" """))

# This is our server. FastAPI creates two endpoints, one dummy ("/") and
# one for serving our prediction ("/{style}"). The serving endpoint takes in a name as a URL parameter.
# [We're using nine different trained models to perform style transfer, so the path parameter
# will tell us which model_frontend to choose. The image is accepted as a file over a POST request and
# sent to the inference function. Once the inference is complete, the file is stored on the local
# filesystem and the path is sent as a response.

# Next, add the following config to a new file called backend/config.py:
