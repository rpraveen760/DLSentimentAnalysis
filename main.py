import tensorflow as tf
import tensorflow_hub as hub
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Ensure TensorFlow Hub module is correctly downloaded
print("Downloading Universal Sentence Encoder...")
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("Universal Sentence Encoder loaded successfully!")

# Define the custom object scope for loading TensorFlow Hub layers
custom_objects = {'KerasLayer': hub.KerasLayer}

# Load the trained model with TensorFlow Hub layers recognized
try:
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model("saved_model")
    print("Model loaded successfully from 'saved_model'.")
except Exception as e:
    print(f"Error loading model: {e}")


class Tweet(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Tweet Disaster Detection API"}


@app.post("/predict/")
def predict(tweet: Tweet):
    # Ensure input is correctly formatted
    raw_text = [tweet.text]  # shape=(1,)

    # Get prediction from the model
    prediction = model.predict(raw_text)
    is_disaster = bool(np.round(prediction[0][0]))

    return {
        "tweet": tweet.text,
        "is_disaster": is_disaster,
        "confidence": float(prediction[0][0])
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
