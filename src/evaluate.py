"""
=========================================================
EcoScan AI
Model Evaluation
=========================================================
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import *
from classes import CLASS_NAMES

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

# =====================================================
# LOAD MODEL
# =====================================================

model = load_model(BEST_MODEL)

print("Model Loaded Successfully")
print(BEST_MODEL)

print("\n")

# =====================================================
# LOAD TEST DATASET
# =====================================================

test_datagen = ImageDataGenerator(

    rescale=1./255

)

test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode=CLASS_MODE,

    shuffle=False

)

print("=" * 60)
print("TEST DATASET")
print("=" * 60)

print("Total Images :", test_generator.samples)

print("Classes :", test_generator.class_indices)

print("=" * 60)

# =====================================================
# PREDICT
# =====================================================

print("\nPredicting...")

prediction = model.predict(

    test_generator,

    verbose=1

)

print("Prediction Finished")

# =====================================================
# LABEL
# =====================================================

y_true = test_generator.classes

y_pred = np.argmax(

    prediction,

    axis=1

)

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(

    y_true,

    y_pred

)

precision = precision_score(

    y_true,

    y_pred,

    average="weighted"

)

recall = recall_score(

    y_true,

    y_pred,

    average="weighted"

)

f1 = f1_score(

    y_true,

    y_pred,

    average="weighted"

)

print("\n")

print("=" * 60)

print("OVERALL METRICS")

print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

print("=" * 60)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(

    y_true,

    y_pred

)

print("\n")

print("=" * 60)

print("CONFUSION MATRIX")

print("=" * 60)

print(cm)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

report = classification_report(

    y_true,

    y_pred,

    target_names=CLASS_NAMES,

    digits=4

)

print("\n")

print("=" * 60)

print("CLASSIFICATION REPORT")

print("=" * 60)

print(report)

# =====================================================
# SAVE REPORT
# =====================================================

report_file = REPORT_DIR / "classification_report.txt"

with open(

    report_file,

    "w",

    encoding="utf-8"

) as f:

    f.write("EcoScan AI\n\n")

    f.write("Overall Metrics\n")

    f.write("----------------------\n")

    f.write(f"Accuracy  : {accuracy:.4f}\n")

    f.write(f"Precision : {precision:.4f}\n")

    f.write(f"Recall    : {recall:.4f}\n")

    f.write(f"F1 Score  : {f1:.4f}\n\n")

    f.write(report)

print("\n")

print("Classification Report Saved")

print(report_file)