from django.urls import path
from .views import generate_images, PredictDigitView, display_images

urlpatterns = [
    path('generate-images/', generate_images, name='generate_images'),
    path('predict-digit/', PredictDigitView.as_view(), name='predict_digit'),
    path('', display_images, name='display_images'),
]

