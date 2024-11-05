# Digit Recognizer

This project is a digit recognition application built using Django, a deep learning model, and the MNIST dataset. The application allows users to select or generate images of handwritten digits and predicts the digit based on the provided image.

## Features

- **Digit Prediction:** Users can select an image to predict the corresponding handwritten digit.
- **Image Generation:** Users can generate random images from the MNIST dataset for testing.
- **Deep Learning Model:** The backend model is based on a deep learning architecture trained on the MNIST dataset.
- **REST API:** The project includes API endpoints for image prediction and image generation.

## Tech Stack

- **Backend Framework:** Django 5.0
- **Frontend:** HTML, CSS, JavaScript
- **Deep Learning:** Custom digit recognizer model using NumPy
- **Database:** MNIST dataset as `.csv`
- **Static Files Management:** Django's static files handling

## Installation

Follow these steps to set up the project locally:

### Prerequisites

- Python 3.11 or later
- Virtualenv

### Setup

1. **Run the development server:**
   ```bash
   python devserver.py
   ```

   By default, the server will run at `http://127.0.0.1:8000/`.

## Usage

### Access the Application

1. **Generate and predict images:**
   - Access the homepage at `http://127.0.0.1:8000/`.
   - Tap on any of the generated images to predict the digit.
   - Click on the "Generate" button to load new random images from the MNIST dataset.

2. **API Endpoints:**
   - `GET /api/display-images/` - Fetches random images from the MNIST dataset.
   - `POST /api/predict-digit/` - Predicts the digit for an uploaded image.

## Model Information

- The deep learning model is loaded from `digit_recognizer_dl_model/digit_recognizer_weights.npz`.
- Ensure that the model weights are placed in this directory before running the application.

## License

This project is licensed under the MIT License.

## Acknowledgments

- The MNIST dataset: [Yann LeCun's MNIST database](http://yann.lecun.com/exdb/mnist/)
