import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import numpy as np
from flask import Flask, request, jsonify, render_template
import os

# Load the trained model
MODEL_PATH = "model_6.h5"  # Ensure the model is in the same directory
model = load_model(MODEL_PATH)

# Define the same text vectorization layer used during training
max_vocab_size = 10000  # Adjust based on your training config
sequence_length = 15  # Adjust to match training input size

text_vectorizer = TextVectorization(max_tokens=max_vocab_size, output_mode="int", output_sequence_length=sequence_length)

# Assuming you have trained with a vocabulary, you should reload it
# Replace with your actual vocabulary file or dataset
# text_vectorizer.adapt(training_texts)  

# Flask app setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")  # Create a simple frontend

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON input
    tweet_text = data.get("text", "")
    
    if not tweet_text:
        return jsonify({"error": "No text provided"})
    
    # Preprocess the text
    vectorized_text = text_vectorizer([tweet_text])  # Convert text to tensor
    prediction_prob = model.predict(vectorized_text)[0][0]  # Get probability
    prediction = int(prediction_prob >= 0.5)  # Convert to 0 or 1
    
    return jsonify({
        "tweet": tweet_text,
        "prediction": "Disaster" if prediction == 1 else "Not Disaster",
        "probability": float(prediction_prob)
    })

if __name__ == '__main__':
    app.run(debug=True)
