import base64
import numpy as np
import streamlit as st
from PIL import ImageOps, Image

# sets the background of a Streamlit app to an image specified by the given image file
def set_background(image_file):

    with open(image_file, 'rb') as f:
        img_data = f.read()
    
    b64_enconded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_enconded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


# takes an image, model and a list of class names and returns the predicted class and confidence score
def classify(image, model, class_names):

    # coonvert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = image_array.astype(np.float32) / 127.5 - 1  # from 0 to 2-> -1 to 1

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)
    print(prediction)
    # index = np.argmax(prediction)
    index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score