import uvicorn
from fastapi import FastAPI, APIRouter, UploadFile, File
import numpy as np
import pandas as pd 

import config
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import spacy 

import warnings 

from schemas import inferance_payload
app = FastAPI()
controller = APIRouter()

preloaded_models = {}
@app.on_event("startup")
def startup_event():
    print("downloading bert sentence transformer")
    preloaded_models['bert'] = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    preloaded_models['keybert'] = KeyBERT('distilbert-base-nli-mean-tokens')
    
    #preloaded_models['spacy_ner'] = spacy.load('en_core_web_sm') 
    warnings.filterwarnings("ignore", category=UserWarning)

@controller.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@controller.post("/inferance")
async def read_text(payload: inferance_payload):
    text = payload.session_note

    extracted_keywords = preloaded_models['keybert'].extract_keywords(text, stop_words='english')
    return {"keywords" : extracted_keywords}


app.include_router(controller)