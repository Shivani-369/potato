from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

MODEL_PATH = 'best_model.h5'
CLASS_NAMES = ['blackheart', 'hollowheart', 'healthy', 'necrosis']

model = load_model(MODEL_PATH)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    print(f"Predicted class: {CLASS_NAMES[class_idx]}")

# Example usage:
# predict_image('data/tubers/healthy/potato1.jpg')
