
# Environmental Sound Classification using Short-Time Hartley Transform (STHT)

This project performs audio classification on environmental sounds (chainsaw, vehicle, speech, forest) using a custom implementation of the Short-Time Hartley Transform (STHT) and K-Nearest Neighbors (KNN) for classification.

## 🔧 Features

- Frame-level feature extraction using STHT
- Low-level features: energy, spectral centroid, roll-off, zero-crossings
- Classifier: KNN with `manhattan` distance
- CLI progress tracking using `tqdm`
- Audio is split into 1-second clips and processed frame by frame

## 📁 Folder Structure

```
.
├── raw_audio/            # Place long audio files here
│   ├── chainsaw_long.wav
│   ├── vehicle_long.wav
│   ├── speech_long.wav
│   └── forest_long.wav
├── dataset_clips/        # Auto-generated audio clips for training
├── index.py              # Main script
└── README.md             # You're here
```

## 📅 Dataset

Download the raw audio files from the following source and place them in the `raw_audio` directory:

**🔗 Download Dataset:**\
[🔊 From my Google Drive](https://drive.google.com/drive/folders/19mlv4eU4-sp6u5rupRU_nhkAV88TKZWU?usp=sharing)

- You may use `chainsaw`, `engine`, `speech`, and `forest` classes to generate long audio clips for this project.
- Alternatively, you can create your own `*_long.wav` files by concatenating multiple 1-second clips using tools like Audacity or ffmpeg.

## 🚀 Getting Started

1. Clone the repository:

```bash
git clone https://github.com/yourusername/knn--audio-training.git
cd knn--audio-training
```

2. Run the script:

```bash
python index.py
```

## ✅ Requirements

- Python 3.7+
- librosa
- soundfile
- numpy
- scikit-learn
- tqdm

Install all using:

```bash
pip install librosa soundfile numpy scikit-learn tqdm
```

##
