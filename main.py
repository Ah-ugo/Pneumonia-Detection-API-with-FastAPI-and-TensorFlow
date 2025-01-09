from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras_preprocessing.image import ImageDataGenerator

app = FastAPI()

# Global variables
labels = ['NORMAL', 'PNEUMONIA']
model = None

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
    global model

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
        Conv2D(64, (3, 3), activation='relu', strides=1, padding='same'),
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

    # Class weights for handling class imbalance
    class_weights = {0: 1.0, 1: 2.0}  # Adjust class weights if needed

    # Train the model
    print("Training the model...")
    model.fit(datagen.flow(train_x, train_y, batch_size=32), epochs=12, validation_data=(val_x, val_y), class_weight=class_weights)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_x, test_y)
    print("Test Accuracy:", test_acc)

    # Save the model for later use
    model.save("pneumonia_model.h5")
    print("Model training complete and saved as 'pneumonia_model.h5'")


def preprocess_image(image_file: UploadFile):
    """Preprocess uploaded image for model prediction."""
    contents = image_file.file.read()
    np_image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Invalid image file")
    image = cv2.resize(image, (100, 100))
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0) / 255.0
    return image


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Predict the class of the uploaded image."""
    global model
    if not model:
        return JSONResponse(status_code=500, content={"message": "Model not loaded"})

    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        print(f"Prediction raw output: {prediction}")  # Debugging line to check prediction value

        # Adjust threshold logic (increase threshold if necessary)
        predicted_class = labels[0 if prediction[0] < 0.7 else 1]
        confidence = float(prediction[0])

        return {"predicted_class": predicted_class, "confidence": confidence}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the Pneumonia Detection API"}
