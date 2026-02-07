import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

PORT = int(os.environ.get("PORT", 8501))

# Load your trained model
model = tf.keras.models.load_model('my_model.h5', compile=False)

# Define categories
categories = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "No Finding"]

# Function to make predictions
def predict_anomaly(image):
  # Return the prediction array
    image = image.convert('RGB')  # Ensure the image has 3 channels
    img = image.resize((128,128))  # Resize the image to (128,128)
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the model's input shape
    predictions = model.predict(img_array)  # Get predictions
    print(f"Predictions: {predictions}")  # Print the prediction array for debugging
    return predictions[0]

st.title("X-Ray Anomaly Detection")
st.write("Upload an X-ray image to detect anomalies.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray.', use_column_width=True)
    st.write("Classifying...")
    # Make prediction
    prediction = predict_anomaly(image)
    # Ensure predictions and categories match in length
    if len(prediction) > len(categories):
        prediction = prediction[:len(categories)]  # Limit predictions to match categories

# Get the top 3 predictions
    top_indices = np.argsort(prediction)[-3:][::-1]  # Get indices of top 3 predictions in descending order
    top_diseases = [(categories[i], prediction[i] * 100) for i in top_indices]  # Get disease names and probabilities

# Display top 3 predictions
    st.write("Top 3 Likely Diagnoses:")
    for disease, prob in top_diseases:
        st.write(f"{disease}: {prob:.2f}%")

# Display the most likely diagnosis
    st.write(f"Most likely diagnosis: {top_diseases[0][0]} with confidence {top_diseases[0][1]:.2f}%")



