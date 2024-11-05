import os
import io
import numpy as np
import pandas as pd
import base64
from PIL import Image
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from rest_framework.views import APIView
from .digit_recognizer import DigitRecognizer
from django.core.files.uploadedfile import InMemoryUploadedFile

class PredictDigitView(APIView):
    model = DigitRecognizer()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_path = os.path.join(settings.BASE_DIR, 'digit_recognizer_dl_model', 'digit_recognizer_weights.npz')
        
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model.load_model(model_path)

    def post(self, request):
        file = request.FILES.get('file')
        if not file or not isinstance(file, InMemoryUploadedFile):
            return JsonResponse({'error': 'No valid file uploaded'}, status=400)

        try:
            image = Image.open(file).convert('L')

            desired_size = 28
            image.thumbnail((desired_size, desired_size), Image.LANCZOS)

            new_image = Image.new('L', (desired_size, desired_size), color=255)
            new_image.paste(
                image, 
                ((desired_size - image.size[0]) // 2, (desired_size - image.size[1]) // 2)
            )

            image_array = np.array(new_image).reshape(784, 1)
            image_array = image_array / 255.0

            prediction = self.model.predict(image_array)
            return JsonResponse({'prediction': int(prediction[0])})

        except Exception as e:
            print(f"Error in predict: {e}")
            return JsonResponse({'error': str(e)}, status=500)

def generate_images(request):
    data_path = os.path.join(settings.BASE_DIR, 'digit_recognizer_dl_model', 'mnist_test.csv')
    data = pd.read_csv(data_path).to_numpy()

    np.random.shuffle(data)

    selected_data = data[:20]
    images = []
    for row in selected_data:
        label = int(row[0])
        pixels = row[1:]

        if len(pixels) < 784:
            pixels = np.pad(pixels, (0, 784 - len(pixels)), mode='constant')
        elif len(pixels) > 784:
            return JsonResponse({'error': f'Unexpected pixel count: {len(pixels)}'}, status=500)

        pixels = pixels.reshape(28, 28)

        image = Image.fromarray(pixels.astype(np.uint8), mode='L')

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        images.append({
            'label': label,
            'image': image_base64
        })

    return JsonResponse({'images': images})

def display_images(request):
    return render(request, 'display_images.html')
