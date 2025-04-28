import cv2
import os

! pip install matplotlib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'Dataset 1\archive\train' 
test_dir = 'Dataset 1\archive\test'

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    print(f"Created directory: {train_dir}")
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
    print(f"Created directory: {test_dir}")


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize ImageDataGenerator for preprocessing
train_data_gen = ImageDataGenerator(rescale=1.0 / 255)
validation_data_gen = ImageDataGenerator(rescale=1.0 / 255)

# Load training data
train_generator = train_data_gen.flow_from_directory(
    train_dir,  # Path to training dataset
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",  # Set to "rgb" for color images
    class_mode='categorical'
)

# Load validation data
validation_generator = validation_data_gen.flow_from_directory(
    test_dir,  # Path to validation/test dataset
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

 # Define and compile the model
emotion_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # Use the correct number of classes
])

# Compile the model
emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
emotion_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

test_loss, test_accuracy = emotion_model.evaluate(validation_generator)
print(f'Test accuracy: {test_accuracy}')

# Save the trained model to a file
emotion_model.save('emotion_model.h5')

# Save model architecture
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights with the correct filename format
# emotion_model.save_weights('emotion_model.weights.h5')  # Use the correct extension

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the emotion detection model
model = load_model('emotion_model.h5')

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def display_frame(frame):
    """Displays the frame using Matplotlib."""
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.001)
    plt.clf()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract and preprocess face region
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = roi_gray / 255.0

        # Predict emotion
        emotion_prediction = model.predict(roi_gray)
        max_index = np.argmax(emotion_prediction[0])

        # Display emotion text
        cv2.putText(frame, emotion_dict[max_index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame using Matplotlib
    display_frame(frame)

# Release webcam
cap.release()
