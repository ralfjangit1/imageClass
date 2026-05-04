import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Keras model
from tensorflow.keras.models import load_model
model = load_model("my_model.keras")  # or .keras

# Class names
class_names = ["cat", "dog"]

IMG_SIZE = (224, 224)

st.title("Cats vs Dogs Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prob = model.predict(img_array)[0][0]

    label = 1 if prob > 0.5 else 0

    st.write("### Prediction:", class_names[label])
    st.write("### Confidence:", float(prob) if label == 1 else float(1 - prob))