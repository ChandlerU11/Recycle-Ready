import json
from PIL import Image
import os

import streamlit as st
import pandas as pd
import numpy as np

import clip
import torch
import pickle


def load_model():
    """Retrieves the trained model and maps it to the CPU by default,
    can also specify GPU here."""
    model = pickle.load(open( "models/recycle_log_reg.pkl", "rb" ))
    return model

def get_image_features(img):
    image_vectors = []
    image = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.encode_image(image)
        image_vectors.append(feats)
    return image_vectors

def predict(img):
    img_vec = get_image_features(img)
    classifier = load_model()
    test = torch.cat(img_vec).cpu().numpy()
    pred = classifier.predict(test)
    return pred

st.title(':earth_americas: Recycle Ready Machine')
instructions = """
    Upload a photo. 
    """
st.write(instructions)

file = st.file_uploader('Upload An Image')
dtype_file_structure_mapping = {
    'All Images': 'consolidated',
    'Images Used To Train The Model': 'train',
    'Images Used To Tune The Model': 'valid',
    'Images The Model Has Never Seen': 'test'
}
device = "cpu"
model, preprocess = clip.load('ViT-B/32', device = device)

if file:  # if user uploaded file
    img = Image.open(file)
    prediction = predict(img)
    st.title("Here is the image you've selected")
    resized_image = img.resize((336, 336))
    st.image(resized_image)
    st.caption(prediction)