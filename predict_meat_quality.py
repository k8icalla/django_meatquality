from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Define paths
model_path = 'datasets/model/meat_quality_model.h5'  # Path to the trained model
new_image_path = 'new_images/sample_image.jpg'  # Path to the new image to be tested

# Load the model
model = load_model(model_path)

# Check if the new image file exists
if not os.path.exists(new_image_path):
    print("Error: The specified image file does not exist!")
else:
    # Load and preprocess the image
    img = load_img(new_image_path, target_size=(224, 224))  # Ensure size matches the model input
    img_array = img_to_array(img) / 255.0  # Normalize the pixel values
    img_array = img_array.reshape((1, 224, 224, 3))  # Add batch dimension

    # Predict the class
    prediction = model.predict(img_array)
    class_index = prediction.argmax()

    # Map class index to labels
    class_labels = {0: "Fresh", 1: "Half Fresh", 2: "Spoiled"}
    predicted_label = class_labels[class_index]

    # Print the prediction
    print("Predicted Class:", predicted_label)
