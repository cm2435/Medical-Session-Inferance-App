# frontend/main.py

import requests
import streamlit as st
from pydantic import BaseModel
from typing import Optional

import json 

STYLES = {
    "Abby": "Abby",
    "Tommy": "Tommy",
    "Sophie": "Sophie",
    "Kirsty": "Kirsty",
    "Ari": "Ari",
}

# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("KiActive ML Web App Demo")

session_note_in = st.text_input("Session Note input", max_chars=512)

# displays a file uploader widget
image = st.file_uploader("Upload an audio file")

# displays the select widget for the styles
practitioner = st.selectbox("Choose the style", [i for i in STYLES.keys()])

# displays a button
payload = {"user_name" : "Tommy Walker",
           "session_number" : 1,
           "session_note" : session_note_in}

if st.button("Make Text Predictions"):
    if session_note_in:
        #files = {"file": image.getvalue()}
        res = requests.post(f"http://backend:8080/inferance", json = payload)
        img_path = res.json()

        #image = Image.open(img_path.get("name"))
        #st.image(image, width=500)
    
        st.write(payload['session_note'])
        st.write(img_path)