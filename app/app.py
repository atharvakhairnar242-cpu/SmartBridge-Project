from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained model
model = None

# Image size required for MobileNetV2
IMG_SIZE = 224

# Class labels (example – change according to dataset)
class_names = [
"Apple___Apple_scab",
"Apple___Black_rot",
"Apple___Cedar_apple_rust",
"Apple___healthy",
"Tomato___Early_blight",
"Tomato___Late_blight",
"Tomato___healthy"
]

# Home page
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


# About page
@app.route("/about")
def about():
    return render_template("about.html")


# Upload page
@app.route("/upload")
def upload():
    return render_template("upload.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    if file:
        result = "Tomato Leaf - Healthy"

        return render_template("result.html", prediction=result)

    return "No image uploaded"


if __name__ == "__main__":
    app.run(debug=True)