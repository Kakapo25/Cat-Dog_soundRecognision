# Import required libraries
import os
import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


# Load audio files and extract features
def extract_features(file_paths, max_pad_len=100):   
    features = []
    labels = []
    for file_path in file_paths:
        sound, sr = sf.read(file_path)  # Load audio file
        mfccs = librosa.feature.mfcc(y=sound, sr=sr)  # Extract MFCC features

        # Pad MFCC features to max_pad_len
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        features.append(mfccs)
        labels.append(os.path.basename(os.path.dirname(file_path)))  # Extract labels (cats or dogs)
    return features, labels


#for cleaner view in terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Set file paths for cat sounds
#fill in path to directory of your training .wav files for cat sounds
cat_dir = ""
cat_sounds = [cat_dir + f for f in os.listdir(cat_dir) if f.endswith('.wav')]

# Set file paths for dog sounds
#fill in path to directory of your training .wav files for dog sounds
dog_dir = ""
dog_sounds = [dog_dir + f for f in os.listdir(dog_dir) if f.endswith('.wav')]

# Combine file paths
file_paths = cat_sounds + dog_sounds

# Extract features and labels
features, labels = extract_features(file_paths)

# Encode labels
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Convert to array and reshape
X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Create a CNN model
print("Creating CNN model\n")
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

# Save the model
model.save("cat_dog_sound_classifier.h5")


