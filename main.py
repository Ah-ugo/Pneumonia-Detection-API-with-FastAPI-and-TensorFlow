from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras_preprocessing.image import ImageDataGenerator
import io
import base64
from starlette.responses import Response

app = FastAPI()

# Global variables
labels = ['NORMAL', 'PNEUMONIA']
model = None
train_history = None

# Paths for data
train_path = './train'
test_path = './test'
val_path = './val'


def get_data(path):
    """Load and preprocess images from the dataset."""
    x = []
    y = []
    for label in labels:
        images_path = os.path.join(path, label)
        if not os.path.exists(images_path):
            print(f"Path does not exist: {images_path}")
            continue
        for image in os.listdir(images_path):
            img_path = os.path.join(images_path, image)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            img = cv2.resize(img, (100, 100))
            x.append(img)
            y.append(0 if label == 'NORMAL' else 1)
    x = np.array(x) / 255.0
    x = np.expand_dims(x, -1)  # Add channel dimension
    y = np.array(y)
    return x, y


@app.on_event("startup")
async def create_and_train_model():
    """Create, train, and save the CNN model on startup."""
    global model, train_history

    # Load datasets
    print("Loading data...")
    train_x, train_y = get_data(train_path)
    val_x, val_y = get_data(val_path)
    test_x, test_y = get_data(test_path)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    datagen.fit(train_x)

    # Define the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', strides=1, padding='same', input_shape=(100, 100, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=2, padding='same'),
        Conv2D(64, (3, 3), activation='relu', strides=1, padding='same'),
        Dropout(0.1),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=2, padding='same'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    print("Training the model...")
    train_history = model.fit(datagen.flow(train_x, train_y, batch_size=32), epochs=12, validation_data=(val_x, val_y))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_x, test_y)
    print("Test Accuracy:", test_acc)

    # Save the model
    model.save("pneumonia_model.h5")
    print("Model training complete and saved as 'pneumonia_model.h5'")


def plot_training_history():
    """Plot training accuracy and loss over epochs."""
    if not train_history:
        return JSONResponse(status_code=500, content={"message": "Training history not available"})

    acc = train_history.history['accuracy']
    val_acc = train_history.history['val_accuracy']
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(epochs, acc, 'b', label='Training Accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation Accuracy')
    ax[0].set_title('Training and Validation Accuracy')
    ax[0].legend()

    ax[1].plot(epochs, loss, 'b', label='Training Loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation Loss')
    ax[1].set_title('Training and Validation Loss')
    ax[1].legend()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    return Response(img_buf.read(), media_type="image/png")


@app.get("/training-history")
async def get_training_history():
    return plot_training_history()


@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the Pneumonia Detection API with Charts"}
