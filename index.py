import os
import numpy as np
import pandas as pd
from collections import Counter
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
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
        norm_frame = abs_frame / np.sum(abs_frame)

        energy = np.sum(abs_frame ** 2)
        centroid = np.sum(np.arange(len(abs_frame)) * abs_frame) / np.sum(abs_frame)
        spread = np.sqrt(np.sum(((np.arange(len(abs_frame)) - centroid) ** 2) * abs_frame) / np.sum(abs_frame))
        rolloff = np.argmax(np.cumsum(abs_frame) >= 0.85 * np.sum(abs_frame))
        zero_crossings = np.count_nonzero(np.diff(np.sign(frame)))
        flatness = np.exp(np.mean(np.log(abs_frame))) / np.mean(abs_frame)
        entropy = -np.sum(norm_frame * np.log2(norm_frame))
        skewness = (np.sum(((np.arange(len(abs_frame)) - centroid) ** 3) * abs_frame) / np.sum(abs_frame)) / (spread ** 3)
        slope = np.polyfit(np.arange(len(abs_frame)), abs_frame, 1)[0]

        features.append([energy, centroid, spread, rolloff, zero_crossings, flatness, entropy, skewness, slope])
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


# ---------- Spectrogram plot from Folders ----------
def plot_sht_spectrogram(signal, sr, label, frame_size=512, hop_size=256):
    spectrogram = compute_stht(signal, frame_size=frame_size, hop_size=hop_size)
    log_spec = np.log1p(np.abs(spectrogram))  # Log scaling

    num_frames = log_spec.shape[1]
    duration = len(signal) / sr
    time_axis = np.arange(num_frames) * hop_size / sr  # time in seconds

    plt.imshow(log_spec, aspect='auto', origin='lower', cmap='viridis',
               extent=[time_axis[0], time_axis[-1], 0, frame_size])
    plt.title(f'STHT Spectrogram - {label}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency Bin')


def visualize_all_classes(data_dir):
    classes = sorted(os.listdir(data_dir))
    num_classes = len(classes)

    plt.figure(figsize=(4 * num_classes, 5))  # Wider, shorter

    for i, label in enumerate(classes):
        class_path = os.path.join(data_dir, label)
        files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        if not files:
            continue
        file_path = os.path.join(class_path, files[0])
        signal, sr = librosa.load(file_path, sr=16000)

        plt.subplot(1, num_classes, i + 1)
        plot_sht_spectrogram(signal, sr, label)

    plt.tight_layout()
    plt.show()




def predict_new_audio(audio_path, knn_model, scaler, label_encoder, clip_length=1.0, sr=16000):
    # Load and split the audio into clips (if long)
    signal, _ = librosa.load(audio_path, sr=sr)
    clip_samples = int(clip_length * sr)
    num_clips = len(signal) // clip_samples

    predictions = []
    for i in range(num_clips):
        start = i * clip_samples
        end = start + clip_samples
        clip = signal[start:end]

        # Extract features (same as training)
        spectrogram = compute_stht(clip)
        features = extract_llf(spectrogram)

        # Scale features (using the same scaler from training)
        features_scaled = scaler.transform([features])

        # Predict
        pred = knn_model.predict(features_scaled)
        pred_label = label_encoder.inverse_transform(pred)[0]
        predictions.append(pred_label)

    return predictions


# ---------- Main Execution ----------
if __name__ == '__main__':
    # Paths
    raw_audio_dir = 'raw_audio'
    data_directory = 'dataset_clips'
    audio_files = {
        'chainsaw': 'chainsaw_long.mp3',
        'handsaw' : 'handsaw_long.mp3',
        'vehicle': 'vehicle_long.mp3',
        'speech': 'speech_long.mp3',
        'forest': 'forest_long.mp3'
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
    knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
    knn.fit(X_train, y_train)
    
    # Evaluate
    y_pred = knn.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred)))

    visualize_all_classes(data_directory)



    # Test a new audio file
    new_audio_path = './test.mp3'  # Replace with your file
    print(f"\nTesting new audio: {new_audio_path}")
    predictions = predict_new_audio(new_audio_path, knn, scaler, label_encoder)

    # Convert to a DataFrame for tabular display
    df = pd.DataFrame({
        "Clip Number": range(1, len(predictions)+1),
        "Predicted Label": predictions
    })

    # Calculate majority vote
    majority_vote = Counter(predictions).most_common(1)[0][0]
    counts = Counter(predictions)

    # Print results
    print("=== Predictions for Each Clip ===")
    print(df.to_string(index=False))

    print("\n=== Summary Statistics ===")
    print(f"Total clips analyzed: {len(predictions)}")
    for label, count in counts.items():
        print(f"{label}: {count} clips ({count/len(predictions)*100:.1f}%)")

    print(f"\nMajority vote (final prediction): {majority_vote}")
