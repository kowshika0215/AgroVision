import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img_array = img.astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0), img

def predict_disease(image_path, model_path, class_names):
    model = load_model(model_path)
    processed, original = preprocess_image(image_path)
    prediction = model.predict(processed)[0]
    class_id = np.argmax(prediction)
    confidence = prediction[class_id]

    label = f"{class_names[class_id]} ({confidence*100:.2f}%)"
    cv2.putText(original, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return original, label