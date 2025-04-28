!pip install librosa scikit-learn numpy pandas
import librosa
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

# Path to dataset (adjust if not using Google Drive/Colab)
DATASET_PATH = '/content/drive/MyDrive/SER/ravdess'

# RAVDESS emotion codes mapped to labels
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


def extract_features(file_path):
    """
    Extracts features from an audio file using librosa.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        np.ndarray: A NumPy array containing the extracted features.
                  Returns None if feature extraction fails.
    """
    try:
        # Load the audio file
        signal, sr = librosa.load(file_path, sr=None)  # sr=None to use native sample rate

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0)

        return mfccs_scaled
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


# Load the dataset
X = []  # Features
y = []  # Labels
for actor_dir in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor_dir)
    if os.path.isdir(actor_path):
        for file_name in os.listdir(actor_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(actor_path, file_name)
                # Extract features
                features = extract_features(file_path)

                if features is not None:
                    # Get emotion label from file name
                    emotion_label = emotion_map[file_name.split('.')[0].split('-')[2]]

                    X.append(features)
                    y.append(emotion_label)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Grid search for best SVM parameters
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
model = grid.best_estimator_
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Best Hyperparameters:", grid.best_params_)
print("Test Accuracy:", round(acc * 100, 2), "%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=le.classes_, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
try:
    from google.colab import files
    uploaded = files.upload()  # Upload a .wav file
    test_audio_path = list(uploaded.keys())[0]

    emotion = predict_emotion(test_audio_path, model, scaler, le)
    print("🎧 Predicted Emotion:", emotion)
except ImportError:
    pass
    emotion = predict_emotion(test_audio_path, model, scaler, le)
    print("🎧 Predicted Emotion:", emotion)
except ImportError:
    pass
