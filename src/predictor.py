import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from src.config import (
    BEST_MODEL,
    IMAGE_SIZE
)
from src.classes import (
    CLASS_NAMES,
    CARBON_FACTOR,
    POINT_FACTOR
)

print("=" * 60)
print("Loading AI Model...")
print("=" * 60)

MODEL = load_model(BEST_MODEL)

print("Model Loaded Successfully")
print(BEST_MODEL)
print("=" * 60)


def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = np.array(image)
    image = image.astype(np.float32)
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    return image

def calculate_result(category, weight):
    carbon = CARBON_FACTOR.get(category, 0) * weight
    points = POINT_FACTOR.get(category, 0)
    eco_score = points + int(carbon * 10)
    return carbon, points, eco_score

def predict(image_path, weight=1):
    image = preprocess_image(image_path)
    prediction = MODEL.predict(
        image,
        verbose=0
    )

    index = int(np.argmax(prediction))
    confidence = float(prediction[0][index])
    category = CLASS_NAMES[index]
    carbon, points, eco_score = calculate_result(
        category,
        weight
    )

    return {
        "category": category,
        "confidence": round(confidence * 100, 2),
        "weight": float(weight),
        "carbon_saved": round(carbon, 2),
        "points": points,
        "eco_score": eco_score
    }

def get_classes():
    return CLASS_NAMES

def get_model():
    return MODEL