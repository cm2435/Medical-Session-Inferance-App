import uvicorn
from fastapi import FastAPI, APIRouter, UploadFile, File
import numpy as np
import pandas as pd 

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import spacy 

import warnings 
import pickle 

from schemas import inferance_payload
from ML import InferanceModels

app = FastAPI()
controller = APIRouter()

preloaded_models = {}
@app.on_event("startup")
def startup_event():
    print("downloading bert sentence transformer")
    preloaded_models['model_set'] = InferanceModels()

    warnings.filterwarnings("ignore", category=UserWarning)

@controller.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@controller.post("/inferance")
async def read_text(payload: inferance_payload):
    text = payload.session_note

    extracted_keywords = preloaded_models['model_set'].make_general_nlp_inferance(input_text= text)
    explainability = preloaded_models['model_set'].shap_attribute_sentiment()
    shap_values = explainability(["text"])

    return {"inferance": extracted_keywords,
            "explainability": pickle.dumps(shap_values)
            }

app.include_router(controller)


