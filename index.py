import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------- Hartley Transform ----------
def dht(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    cas = np.cos(2 * np.pi * k * n / N) + np.sin(2 * np.pi * k * n / N)
    return np.dot(cas, x)

# ---------- Short-Time Hartley Transform ----------
def compute_stht(signal, frame_size=512, hop_size=256):
    window = np.hamming(frame_size)
    num_frames = 1 + (len(signal) - frame_size) // hop_size
    spectrogram = np.zeros((frame_size, num_frames))

    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start:start + frame_size] * window
        spectrogram[:, i] = dht(frame)

    return spectrogram

# ---------- Low-Level Feature Extraction ----------
def extract_llf(spectrogram):
    features = []
    for frame in tqdm(spectrogram.T, desc="Extracting features", leave=False):
        abs_frame = np.abs(frame) + 1e-8
        energy = np.sum(abs_frame ** 2)
        centroid = np.sum(np.arange(len(abs_frame)) * abs_frame) / np.sum(abs_frame)
        rolloff = np.argmax(np.cumsum(abs_frame) >= 0.85 * np.sum(abs_frame))
        zero_crossings = np.count_nonzero(np.diff(np.sign(frame)))
        features.append([energy, centroid, rolloff, zero_crossings])
    return np.array(features).flatten()

# ---------- Split Long Audio into Clips ----------
def split_audio_to_clips(audio_path, output_dir, label, clip_length=1.0, sr=16000):
    signal, _ = librosa.load(audio_path, sr=sr)
    clip_samples = int(clip_length * sr)
    num_clips = len(signal) // clip_samples

    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(num_clips), desc=f"Splitting '{label}'"):
        start = i * clip_samples
        end = start + clip_samples
        clip = signal[start:end]
        filename = os.path.join(output_dir, f'{label}_{i}.wav')
        sf.write(filename, clip, sr)

# ---------- Load All Clips from Folders ----------
def load_clips_from_folder(data_dir):
    X, y = [], []
    for label in tqdm(os.listdir(data_dir), desc="Loading folders"):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        files = [f for f in os.listdir(label_dir) if f.endswith(".wav")]
        for file in tqdm(files, desc=f"Processing '{label}'", leave=False):
            path = os.path.join(label_dir, file)
            signal, sr = librosa.load(path, sr=16000)
            spectrogram = compute_stht(signal)
            features = extract_llf(spectrogram)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# ---------- Main Execution ----------
if __name__ == '__main__':
    # Paths
    raw_audio_dir = 'raw_audio'
    data_directory = 'dataset_clips'
    audio_files = {
        'chainsaw': 'chainsaw_long.wav',
        'vehicle': 'vehicle_long.wav',
        'speech': 'speech_long.wav',
        'forest': 'forest_long.wav'
    }

    # Create clips if needed
    for label, filename in audio_files.items():
        output_dir = os.path.join(data_directory, label)
        input_path = os.path.join(raw_audio_dir, filename)

        if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
            if not os.path.exists(input_path):
                print(f"Missing file: {input_path}. Please add it to proceed.")
                continue
            print(f"Creating clips for {label} from {input_path}...")
            split_audio_to_clips(input_path, output_dir, label)

    # Load and process dataset
    print("Loading and extracting features from dataset...")
    X, y = load_clips_from_folder(data_directory)

    # Encode string labels into integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Normalize and split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

    # Train KNN classifier with a progress bar (mocked since KNN is fast)
    print("Training KNN classifier...")
    for _ in tqdm(range(1), desc="Training"):
        knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
        knn.fit(X_train, y_train)

    # Evaluate
    y_pred = knn.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred)))
