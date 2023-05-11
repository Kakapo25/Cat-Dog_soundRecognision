import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = tf.keras.models.load_model("cat_dog_sound_classifier.h5")

#for cleaner view in terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Function to extract MFCC features from a single audio file
def extract_mfcc(file_path, max_pad_len=100):
    sound, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=sound, sr=sr)
    
    # Pad MFCC features to max_pad_len
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]

    return mfccs



# Function to predict the label of a single audio file
def predict_label(file_path, model, encoder):
    mfccs = extract_mfcc(file_path)
    mfccs = np.expand_dims(mfccs, axis=-1)  # Add channel dimension
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    prediction = model.predict(mfccs)
    predicted_label = encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_label

# Define the file paths for the new audio files
#fill in path to directory of your testing .wav files for cat sounds
cat_dir = ""
new_cat_sounds = [cat_dir + f for f in os.listdir(cat_dir) if f.endswith('.wav')]
#fill in path to directory of your testing .wav files for dog sounds
dog_dir = ""
new_dog_sounds = [dog_dir + f for f in os.listdir(dog_dir) if f.endswith('.wav')]


# Create a label encoder (use the same one from the training script if available)
encoder = LabelEncoder()
encoder.fit(["cats", "dogs"])

# Initialize counters for correct and total predictions
correct_predictions = 0
total_predictions = 0

# Test the model on the new audio files
for file_path in new_cat_sounds + new_dog_sounds:
    predicted_label = predict_label(file_path, model, encoder)
    if (("cat_" in file_path) and (predicted_label == "cats")) or (("dog_" in file_path) and (predicted_label == "dogs")):
        correct_predictions += 1    
    total_predictions += 1
    print(f"File: {file_path} - Predicted Label: {predicted_label}")

# Calculate and print the accuracy
accuracy = (correct_predictions / total_predictions) * 100
print("Correct predictions: " + str(correct_predictions) + "\n Incorrect: " + str(total_predictions-correct_predictions))
print(f"Accuracy: {accuracy:.2f}%")
