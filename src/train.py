"""
=========================================================
EcoScan AI
Training Model (Part 3A)
Transfer Learning MobileNetV2
=========================================================
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)

from config import *
from classes import CLASS_NAMES


# =====================================================
# GPU INFO
# =====================================================

print("=" * 60)
print("TensorFlow Version :", tf.__version__)
print("=" * 60)

print("GPU Available :", len(tf.config.list_physical_devices('GPU')))


# =====================================================
# DATA AUGMENTATION
# =====================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.20,
    horizontal_flip=True,
    fill_mode="nearest"
)

validation_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)


# =====================================================
# DATASET
# =====================================================

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    shuffle=False
)

print("\n")
print("=" * 60)
print("CLASS INDEX")
print("=" * 60)
print(train_generator.class_indices)
print("=" * 60)


# =====================================================
# LOAD PRETRAINED MODEL
# =====================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# =====================================================
# FREEZE LAYER
# =====================================================

for layer in base_model.layers:
    layer.trainable = False

print("\n")
print("Base Model Loaded")
print("Frozen Layer :", len(base_model.layers))


# =====================================================
# BUILD MODEL
# =====================================================

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)


# =====================================================
# COMPILE
# =====================================================

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

print("\n")
print("=" * 60)
print("MODEL SUMMARY")
print("=" * 60)
model.summary()


# =====================================================
# CALLBACKS
# =====================================================

# Membuat folder jika belum ada
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=str(BEST_MODEL),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    verbose=1,
    min_lr=1e-7
)

csv_logger = CSVLogger(
    REPORT_DIR / "training_log.csv",
    append=False
)

callbacks = [
    checkpoint,
    early_stopping,
    reduce_lr,
    csv_logger
]

print("\n")
print("=" * 60)
print("CALLBACK READY")
print("=" * 60)

print("\n")
print("=" * 60)
print("Training Dataset   :", train_generator.samples)
print("Validation Dataset :", validation_generator.samples)
print("Testing Dataset    :", test_generator.samples)
print("=" * 60)


# =====================================================
# MODEL TRAINING
# =====================================================

print("\n")
print("=" * 60)
print("START TRAINING")
print("=" * 60)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks
)


# =====================================================
# PLOT TRAINING HISTORY
# =====================================================

plt.figure(figsize=(10, 6))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(REPORT_DIR / "training_accuracy.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(REPORT_DIR / "training_loss.png")
plt.close()

print("Accuracy & Loss Graph Saved")


# =====================================================
# PRECISION GRAPH
# =====================================================

if "precision" in history.history:
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["precision"], label="Train Precision")
    plt.plot(history.history["val_precision"], label="Validation Precision")
    plt.title("Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.savefig(REPORT_DIR / "precision.png")
    plt.close()


# =====================================================
# RECALL GRAPH
# =====================================================

if "recall" in history.history:
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["recall"], label="Train Recall")
    plt.plot(history.history["val_recall"], label="Validation Recall")
    plt.title("Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid(True)
    plt.savefig(REPORT_DIR / "recall.png")
    plt.close()

print("Precision & Recall Graph Saved")


# =====================================================
# TEST DATASET EVALUATION
# =====================================================

print("\n")
print("=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

test_result = model.evaluate(
    test_generator,
    verbose=1
)

print("\n")
print("=" * 60)
print("TEST RESULT")
print("=" * 60)
print(f"Loss       : {test_result[0]:.4f}")
print(f"Accuracy   : {test_result[1]*100:.2f}%")
print(f"Precision  : {test_result[2]*100:.2f}%")
print(f"Recall     : {test_result[3]*100:.2f}%")
print("=" * 60)


# =====================================================
# SAVE FINAL MODEL & TRAINING COMPLETE
# =====================================================

model.save(FINAL_MODEL)

print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Best Model : {BEST_MODEL}")
print(f"Final Model: {FINAL_MODEL}")
print(f"Reports    : {REPORT_DIR}")
print("=" * 60)