import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from django.shortcuts import render
from .forms import ImageUploadForm
import os

MODEL_PATH = './datasets/model/meat_quality_model.h5'
IMG_HEIGHT, IMG_WIDTH = 150, 150

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ['Fresh', 'Half Fresh', 'Spoiled']

def predict_quality(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension

    # Predict
    predictions = model.predict(img_array)
    class_idx = tf.argmax(predictions[0])
    return CLASS_NAMES[class_idx]

def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_path = os.path.join('media', image.name)

            # Save the uploaded image
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Predict quality
            prediction = predict_quality(image_path)
            return render(request, 'index.html', {'prediction': prediction, 'image_url': image_path})

    else:
        form = ImageUploadForm()

    return render(request, 'index.html', {'form': form})
