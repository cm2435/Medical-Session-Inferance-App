import torch
import transformers
from transformers import BertForSequenceClassification, BertTokenizer, Trainer
import numpy as np

import spacy
import pickle 
import pathlib
from keybert import KeyBERT
import torch
import pandas as pd
import shap
import scipy as sp

from io import BytesIO
class Dataset(torch.utils.data.Dataset):
    #DataLoader Class for Transformer Inferance
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class SentimentTransformer(object):
    def __init__(self):
        #FOR DOCKER USE THESE
        self.sentiment_transformer = BertForSequenceClassification.from_pretrained(r"/app/models/sentiment_model")
        self.tokenizer = BertTokenizer.from_pretrained(r"/app/models/sentiment_tokenizer")

        #FOR LOCAL USE THESE
        #self.sentiment_transformer = BertForSequenceClassification.from_pretrained(r"backend\models\sentiment_model")
        #self.tokenizer = BertTokenizer.from_pretrained(r"backend\models\sentiment_tokenizer")
        self.softmax = torch.nn.Softmax(dim=1)

    def predict_sentiment(self, input_sentence: str):
        encoded_input = self.tokenizer(input_sentence, truncation=True, padding=True, return_tensors="pt")
        dataclass = Dataset(encoded_input)
        trainer = Trainer(self.sentiment_transformer)
        raw_pred, _, _ = trainer.predict(dataclass)
        
        # Preprocess raw predictions
        softmaxed_preds = self.softmax(torch.tensor(raw_pred)).tolist()[0]
        labels = ["negative", "neutral", "positive", "very_positive"]
        return {k:v for k,v in list(zip(labels,softmaxed_preds))}


class InferanceModels(SentimentTransformer):
    def __init__(self):
        super().__init__()
        self.fp = "\app\models"
        self.NER = spacy.load('en_core_web_sm')
        self.keyphrase_model = KeyBERT('distilbert-base-nli-mean-tokens')

    def make_NER_prediction(self, input_text: str) -> dict:
        NER_prediction = self.NER(input_text).ents
        return{k:v for k,v in list(zip([ent.text for ent in NER_prediction], [ent.label_ for ent in NER_prediction]))}

    def make_keyphrase_prediction(self, input_text):
        return self.keyphrase_model.extract_keywords(input_text, stop_words='english')
        
    def make_general_nlp_inferance(self, input_text: str):
        #Make numeric inferances 
        sentiment_prediction =  self.predict_sentiment(input_text)
        ner_prediction = self.make_NER_prediction(input_text)
        keyphrase_prediction = self.make_keyphrase_prediction(input_text)

        #Get attribution for sentiment
        #infer = self.shap_attribute_sentiment()

        return {"sentiment_prediction": sentiment_prediction,
                "NER_prediction" : ner_prediction,
                "extracted_keywords" : keyphrase_prediction,
                #"shap_values": infer([input_text])    
                }
    
    def f(self, x):
        tv = torch.tensor([self.tokenizer.encode(v, padding='max_length', max_length=128,truncation=True) for v in x])
        attention_mask = (tv!=0).type(torch.int64)
        outputs = self.sentiment_transformer(tv,attention_mask=attention_mask)[0].detach().numpy() #.cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores) 
        return val

    def shap_attribute_sentiment(self):
        labels = ["negative", "neutral", "positive", "v_positive"]
        explainer = shap.Explainer(self.f, self.tokenizer,output_names=labels)
        return explainer

"""infer = InferanceModels().shap_attribute_sentiment()
shap_values = infer(["I want to jump off a fucking bridge ugh"])
print("shape",shap_values.shape)

print(str(shap_values))

print(type(shap_values))
print(pickle.dumps(shap_values))
#print(bytes(shap.plots.bar(shap_values[:,:,"neutral"].mean(0), order=shap.Explanation.argsort)))
"""