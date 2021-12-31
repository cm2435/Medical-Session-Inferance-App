# frontend/main.py

import requests
import streamlit as st

from streamlit import session_state
from pydantic import BaseModel
from typing import Optional
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px

import json 
import shap

#when we import hydralit, we automatically get all of Streamlit
import hydralit as hy

#custom user session parameters to share across apps
user_s_params = {'session_id': 42, 'my_special_value': 'Hydralit Fun!'}

app = hy.HydraApp(title='Simple Multi-Page App',navbar_animation=True, session_params=user_s_params,\
favicon="🐙",hide_streamlit_markers=True,use_navbar=True, navbar_sticky=True)
@app.addapp(is_home=True)
def my_home():
    

    STYLES = {
    "Abby": "Abby",
    "Tommy": "Tommy",
    "Sophie": "Sophie",
    "Kirsty": "Kirsty",
    "Ari": "Ari",}
    
    # https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
    st.set_option("deprecation.showfileUploaderEncoding", False)

    # defines an h1 header
    st.title("KiActive ML Web App Demo")

    st.header("Please input the session notes for the desired patient")

    session_note_in_1 = st.text_input("First Session Note input", max_chars=512)
    session_note_in_2 = st.text_input("Second Session Note input", max_chars=512)

    # displays a file uploader widget
    image = st.file_uploader("Upload an audio file")

    # displays the select widget for the styles
    practitioner = st.selectbox("Who presided over session", [i for i in STYLES.keys()])

    # displays a button
    payload = {"user_name" : "Tommy Walker",
            "session_number" : 1,
            "session_note" : session_note_in_1 + session_note_in_2}

    if st.button("Make Text Predictions"):
        if session_note_in_1 and session_note_in_2:
            res = requests.post(f"http://backend:8080/inferance", json = payload)
            st.session_state.api_result = res

            results = res.json()

            st.header("Results")
            st.subheader("Initial Input text")
            st.write(payload['session_note'])
            st.subheader("Top KeyPhrases")
            st.write(results)


            #df = df = pd.DataFrame.from_dict(results)
            
            #The plot
            sentiment_df = pd.DataFrame({"Sentiment" : list(results['sentiment_prediction'].keys()),
                    "Probability of sentiment" : list(results["sentiment_prediction"].values())})

            fig = px.bar(        
            sentiment_df,
            x = "Sentiment",
            y = "Probability of sentiment",
            title = "Sentiment Bar Graph")
            st.plotly_chart(fig)

            sentiment_score = results['sentiment_prediction']['positive'] - results['sentiment_prediction']['negative']
            st.caption('Sentiment Score for patient: {}'.format(sentiment_score))

            shap_values = results['shap_values']
            shap.plots.bar(shap_values[:,:,"neutral"].mean(0), order=shap.Explanation.argsort)
            st.pyplot()

@app.addapp()
def app2():
    hy.info('Hello from app 2')
    app.session_state.my_special_value='I changed the parameter'

    st.write(st.session_state['api_result'])

@app.addapp(title='The Best', icon="🥰")
def app3():
    hy.info('Hello from app 3, A.K.A, The Best 🥰')
    app.session_state.session_id=51

#Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()

#print all the user defined session state parameters and their values





