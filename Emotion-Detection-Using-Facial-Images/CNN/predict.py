import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf

# Load a test image
imag = cv2.imread(os.getcwd() + '/CNN/data/test/sad/PrivateTest_7622844.jpg')
img_from_ar = Image.fromarray(imag, 'RGB')
resized_image = img_from_ar.resize((48, 48))

# Prepare the test image for prediction
test_image = np.expand_dims(np.array(resized_image), axis=0)

# Load the trained model
model = tf.keras.models.load_model(os.getcwd() + '/model.h5')

# Make predictions
result = model.predict(test_image)

# Display prediction results
print("Predicted Probabilities:", result)
predicted_class = np.argmax(result)
print("Predicted Class:", predicted_class)
