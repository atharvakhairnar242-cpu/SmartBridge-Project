# PlantCare AI: Advanced Plant Disease Detection Using Transfer Learning

## Project Overview

PlantCare AI is an intelligent plant disease detection system that uses deep learning and transfer learning with MobileNetV2 to identify diseases in plant leaves from images.

The goal of this project is to help farmers, gardeners, and agricultural experts detect plant diseases quickly and accurately through a web application built with Flask.

The model is trained on plant leaf image datasets and can predict the disease category from a leaf image uploaded by the user.

---

## Objectives

* Build an AI model capable of detecting plant diseases from leaf images.
* Use transfer learning with MobileNetV2 to improve performance.
* Train and evaluate the model using image datasets.
* Deploy the trained model using a Flask web application for real-time predictions.

---

## Technologies Used

* Python
* TensorFlow / Keras
* MobileNetV2 (Transfer Learning)
* Flask
* HTML / CSS
* Matplotlib / Seaborn
* Git & GitHub

---

## Project Structure

```
PlantCare-AI
│
├── data/
│   ├── raw/                 # original dataset
│   └── processed/           # cleaned and augmented dataset
│
├── notebooks/               # data exploration and visualization
│
├── src/
│   ├── data_preprocessing/  # preprocessing and augmentation scripts
│   ├── model_building/      # MobileNetV2 model setup
│   ├── training/            # training pipeline
│   └── evaluation/          # model evaluation
│
├── models/                  # saved trained models
│
├── app/                     # Flask web application
│   ├── app.py
│   └── templates/           # HTML templates
│
├── requirements.txt
└── README.md
```

---

## Project Workflow

1. User opens the web application.
2. User uploads a plant leaf image.
3. The uploaded image is resized to 224x224 and normalized.
4. The MobileNetV2 model analyzes the image.
5. The predicted plant disease class is displayed to the user.

Pipeline:

Dataset → Preprocessing → Model Training → Model Evaluation → Web Application


### Data Collection and Preprocessing

* Download plant disease dataset
* Explore dataset using visualizations
* Apply preprocessing and augmentation techniques

### Model Building

* Load pre-trained MobileNetV2
* Apply transfer learning
* Configure model layers

### Model Training

* Compile the model
* Configure callbacks
* Train the model on the dataset
* Visualize training performance

### Model Evaluation

* Evaluate model accuracy and performance
* Save the final trained model

### Application Development

* Build Flask backend (app.py)
* Create HTML interface for image upload
* Integrate the trained model for predictions

---

## Expected Output

* Trained plant disease detection model
* Performance evaluation metrics
* Web application for plant disease prediction

---

## Future Improvements

* Improve model accuracy with more data
* Deploy the application on cloud platforms
* Add more plant species and diseases
* Develop a mobile application version

---

## Conclusion

PlantCare AI demonstrates the practical use of artificial intelligence and transfer learning in agriculture, providing an accessible tool for early plant disease detection and improved crop health management.
