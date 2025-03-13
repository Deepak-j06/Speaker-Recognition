#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install speechbrain torchaudio torch librosa')


# In[7]:


import torchaudio
import torch
import speechbrain as sb
from speechbrain.inference import SpeakerRecognition
import os


# In[9]:


# Load the pre-trained ECAPA-TDNN model from SpeechBrain
model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa"
)


# In[11]:


data_folder = "17paras"  # Folder where your speaker subfolders exist

# List all speaker directories
speakers = os.listdir(data_folder)
print("Detected speakers:", speakers)


# In[15]:


dataset_path = "17paras"

# Get list of speakers (folder names)
speakers = sorted(os.listdir(dataset_path))
speaker_to_id = {spk: i for i, spk in enumerate(speakers)}

# Load all data
data = []
labels = []

for speaker in speakers:
    speaker_folder = os.path.join(dataset_path, speaker)
    for filename in os.listdir(speaker_folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(speaker_folder, filename)
            data.append(filepath)
            labels.append(speaker_to_id[speaker])

# Convert labels to tensor
labels = torch.tensor(labels)
print(f"Loaded {len(data)} audio files from {len(speakers)} speakers.")


# In[16]:


import torch
from speechbrain.inference import EncoderClassifier
import torchaudio

# Load the pre-trained ECAPA-TDNN model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="speechbrain_pretrained"
)

def extract_embeddings(file):
    """Extract ECAPA-TDNN embeddings and ensure consistent shape."""
    signal = classifier.load_audio(file)  # Load the audio
    embeddings = classifier.encode_batch(signal).detach()  # Get embeddings

    # Ensure shape is always [1, 192]
    embeddings = embeddings.squeeze(0)  # Remove batch dim
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)  # Ensure (1, 192)

    return embeddings


# In[17]:


# Extract features with consistent shape
X = [extract_embeddings(file) for file in data]
X = torch.cat(X, dim=0)  # Concatenate correctly
y = torch.tensor(labels)

print("X Shape:", X.shape)  # Should be (num_samples, 192)
print("y Shape:", y.shape)  # Should be (num_samples,)


# In[43]:


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Create dataset & dataloader
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define classifier
class SpeakerClassifier(nn.Module):
    def __init__(self, input_size=192, num_classes=len(speakers)):
        super(SpeakerClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize model, loss, optimizer
model = SpeakerClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the classifier
for epoch in range(40):  # 20 epochs
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# In[70]:


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Create dataset & dataloader
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define classifier
class SpeakerClassifier(nn.Module):
    def __init__(self, input_size=192, num_classes=len(speakers)):
        super(SpeakerClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize model, loss, optimizer
model = SpeakerClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the classifier
for epoch in range(20):  # 20 epochs
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# In[71]:


print("Speakers List:", speakers)


# In[72]:


def predict_speaker(model, classifier, audio_path):
    # Extract embedding
    embedding = extract_embeddings(audio_path) # Pass only the audio_path
    print("Embedding Shape:", embedding.shape)  # Expected (192,) or (256,)

    # Ensure embedding is in the correct format
    embedding = embedding.unsqueeze(0)  # Shape should be (1, 192) or (1, 256)
    print("Reshaped Embedding Shape:", embedding.shape)

    # Get model output
    with torch.no_grad():
        output = model(embedding)
        print("Model Output Shape:", output.shape)  # Expected (1, num_speakers)

    # Debug: Print full output tensor
    print("Model Raw Output:", output)

    # Get the predicted label
    pred_label = torch.argmax(output, dim=-1).item()
    print("Predicted Label Index:", pred_label)

    # Check if pred_label is valid
    if pred_label >= len(speakers):
        print("Error: Predicted label out of range!")
        return "Unknown"

    return speakers[pred_label]


# In[73]:


test_audio = "17paras/likith/lik_1.wav"
predicted_speaker = predict_speaker(model, classifier, test_audio)
print(f"Predicted Speaker: {predicted_speaker}")


# In[74]:


import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Call the existing function
test_audio = "test/newraj.wav"
predicted_speaker = predict_speaker(model, classifier, test_audio)
print(f"Predicted Speaker: {predicted_speaker}")

# Get model output again for confidence scores
with torch.no_grad():
    embedding = extract_embeddings(test_audio).unsqueeze(0)  # Ensure correct shape
    output = model(embedding)

# Apply softmax to convert logits into probabilities
softmax_probs = F.softmax(output, dim=-1).squeeze()  # Remove extra dimensions if needed
softmax_probs = softmax_probs.cpu().numpy()  # Convert only after calling .cpu()

# Plot confidence scores
plt.figure(figsize=(10, 5))
plt.bar(np.arange(len(speakers)), softmax_probs, color='skyblue')
plt.xticks(np.arange(len(speakers)), speakers, rotation=45)
plt.ylabel("Confidence Score")
plt.xlabel("Speakers")
plt.title(f"Speaker Prediction Confidence - {predicted_speaker}")
plt.ylim(0, 1)  # Confidence scores range from 0 to 1
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Highlight the predicted speaker
pred_label = np.argmax(softmax_probs)
plt.bar(pred_label, softmax_probs[pred_label], color='green', label="Predicted Speaker")
plt.legend()

# Show the plot
plt.show()


# In[108]:


import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Define a threshold for unknown speakers
UNKNOWN_THRESHOLD = 0.3  # Adjust based on performance

def predict_speaker_with_confidence(model, classifier, audio_path):
    """Predict speaker with confidence score comparison."""
    embedding = extract_embeddings(audio_path).unsqueeze(0)  # Ensure correct shape

    with torch.no_grad():
        output = model(embedding)

    # Apply softmax for confidence scores
    softmax_probs = F.softmax(output, dim=-1).squeeze().cpu().numpy()

    # Get the most confident prediction
    pred_label = np.argmax(softmax_probs)
    confidence = softmax_probs[pred_label]

    # Handle unknown speakers
    if confidence < UNKNOWN_THRESHOLD:
        predicted_speaker = "Unknown"
    else:
        predicted_speaker = speakers[pred_label]

    print(f"\n🔍 Predicted Speaker: {predicted_speaker} (Confidence: {confidence:.4f})")

    # Print all confidence scores
    print("\n📊 Confidence Scores for Each Speaker:")
    for i, speaker in enumerate(speakers):
        print(f"{speaker}: {softmax_probs[i]:.4f}")

    # Plot confidence scores
    plt.figure(figsize=(10, 5))
    bars = plt.bar(np.arange(len(speakers)), softmax_probs, color='skyblue')
    plt.xticks(np.arange(len(speakers)), speakers, rotation=45)
    plt.ylabel("Confidence Score")
    plt.xlabel("Speakers")
    plt.title(f"Speaker Prediction Confidence - {predicted_speaker}")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Highlight the predicted speaker
    if predicted_speaker != "Unknown":
        bars[pred_label].set_color('green')  # Change color to green
        plt.text(pred_label, softmax_probs[pred_label] + 0.02, f"{confidence:.2f}",
                 ha='center', fontsize=12, fontweight='bold')

    plt.legend(["Other Speakers", "Predicted Speaker"], loc="upper right")
    plt.show()

    return predicted_speaker, confidence

# Test with an audio file
test_audio = "test17.wav"
predicted_speaker, confidence = predict_speaker_with_confidence(model, classifier, test_audio)



# In[75]:


import torch
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import time

# Function to record audio
def record_audio(filename="realtime.wav", duration=5, sr=16000):
    print("🎤 Recording... Speak Now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("✅ Recording saved as", filename)
    
    
def predict_speaker_realtime(model, classifier, audio_path):
    embedding = extract_embeddings(audio_path).unsqueeze(0).to(device)  # Shape: (1, 192)

    with torch.no_grad():
        outputs = model(embedding)  # Forward pass

        # Debugging: Print output shape
        print("\n🔹 Model Output Shape:", outputs.shape)

        # Ensure softmax is applied to a correct dimension
        softmax_probs = torch.nn.functional.softmax(outputs, dim=-1)  # Convert to probabilities
        
        # Debugging: Check softmax probabilities
        print("\n🔹 Softmax Probabilities Shape:", softmax_probs.shape)
        print("🔹 Softmax Probabilities:", softmax_probs.cpu().numpy())

        # Check if we have multiple speaker classes
        if softmax_probs.shape[1] < 2:
            print("❌ ERROR: Model did not output multiple class scores. Check training setup!")
            return "Unknown", 0.0  # Return default values if something is wrong

        pred_label = torch.argmax(softmax_probs, dim=-1).cpu().item()
        confidence = softmax_probs.squeeze(0)[pred_label].cpu().item() * 100

    print(f"\n🔊 Predicted Speaker: **{speakers[pred_label]}** (Confidence: {confidence:.2f}%)")
    return speakers[pred_label], confidence


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Realtime Testing
record_audio("realtime.wav", duration=5)  # Record a 5-second audio sample
predicted_speaker, confidence = predict_speaker_realtime(model, classifier, "realtime.wav")



# In[76]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Get predictions for all test samples
y_pred = [predict_speaker(model, classifier, file) for file in data]
y_true = [speakers[label] for label in labels.numpy()]

# Compute accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=speakers)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, xticklabels=speakers, yticklabels=speakers, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[77]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[79]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test2.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test2.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test2.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[80]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test3.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test3.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test3.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[81]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test4.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test4.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test4.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[82]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test5.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test5.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test5.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[83]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test6.wav", duration=6, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test6.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test6.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[84]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test7.wav", duration=6, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test7.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test7.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[85]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test8.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test8.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test8.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[86]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test9.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test9.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test9.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[87]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test10.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test10.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test10.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[88]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test11.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test11.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test11.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[89]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test12.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test12.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test12.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[90]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test13.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test13.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test13.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[91]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test14.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test14.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test14.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[92]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test15.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test15.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test15.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[93]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test16.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test16.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test16.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[94]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test17.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test17.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test17.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[95]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test18.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test18.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test18.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[96]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test19.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test19.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test19.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[97]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test20.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test20.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test20.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[98]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test21.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test21.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test21.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[99]:


import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="test22.wav", duration=4, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, sr, audio)
    print("Recording saved.")

# Record and Predict
record_audio("test22.wav", duration=4)
predicted_speaker = predict_speaker(model, classifier, "test22.wav")
print(f"Predicted Speaker: {predicted_speaker}")


# In[109]:


import os
print(os.getcwd())  # Check current directory


# In[ ]:




