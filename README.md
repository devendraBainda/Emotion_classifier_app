# Speech Emotion Classification using Deep Learning

## 📌 Project Objective

The goal of this project is to design and implement an end-to-end pipeline for **emotion classification using speech data**. The system leverages audio processing techniques, data augmentation, feature extraction, and deep learning models to accurately identify and classify emotional states conveyed in speech and song recordings.

---

## 🎯 Key Features

- Audio preprocessing with MFCC, ZCR, RMSE feature extraction
- Data augmentation for improving generalization
- Deep learning model based on 1D CNN architecture
- Streamlit-based web application for real-time emotion prediction
- Supports both single file and batch audio file uploads
- Rich visualizations for confidence scores and waveform analysis

---

## 🗂 Dataset

The dataset used is the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

- Total recordings: **9808 audio samples**
- Emotions classified:
  - Neutral
  - Calm
  - Happy
  - Sad
  - Angry
  - Fearful
  - Disgust
  - Surprised

| Emotion   | Samples |
| --------- | ------- |
| Calm      | 376     |
| Happy     | 376     |
| Sad       | 376     |
| Angry     | 376     |
| Fearful   | 376     |
| Disgust   | 192     |
| Surprised | 192     |
| Neutral   | 188     |

---

## 🔬 Methodology

### 1. Audio Preprocessing

- Sample Rate: 22050 Hz
- Frame Length: 2048
- Hop Length: 512
- Extracted Features:
  - **MFCC (Mel Frequency Cepstral Coefficients)**
  - **Zero Crossing Rate (ZCR)**
  - **Root Mean Square Energy (RMSE)**
- Feature Vector Size: 2376 features

### 2. Data Augmentation

- Time Stretching
- Pitch Shifting
- Adding Noise
- Shifting Audio
- Speed Variation

### 3. Model Architecture

A 1D Convolutional Neural Network (CNN) was designed with the following structure:

- 5 Convolutional layers with batch normalization and max pooling
- Dropout layers for regularization
- Dense layer with 512 neurons
- Output layer with softmax activation (8 emotion classes)
- Total Parameters: \~7.19 Million

### 4. Model Training

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Batch Size: 64
- Epochs: 40
- Learning Rate Scheduler: ReduceLROnPlateau

---

## 📊 Model Performance

- **Overall Test Accuracy:** 93.93%
- **F1 Scores:**
  - Macro F1: 93.86%
  - Weighted F1: 93.94%
  - Micro F1: 93.93%

---

## 🌐 Web Application (Streamlit)

The project includes an interactive web application built using **Streamlit**, which allows users to:

- Upload single or multiple audio files
- Visualize waveform plots
- View confidence scores for each predicted emotion
- Download batch analysis results as CSV
- View pie charts and histograms of predictions

### Web App Features

- 🎯 Real-time inference
- 📈 Confidence visualization (bar & pie charts)
- 📊 Batch processing support
- 🎧 Interactive audio playback
- 🔄 Easy model loading with automatic checks

---

## 📦 Code Structure

- **Notebook (.ipynb):** Full data preparation, feature extraction, model training pipeline.
- **Python App (.py):** Streamlit web app for inference and visualization.
- **Models:**
  - Trained CNN model (`cnn_model_full.keras`)
  - Scaler and Encoder for feature normalization and label encoding.

---

## 🚀 How to Run

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch Streamlit app:

```bash
streamlit run app.py
```

4. Upload your audio files and analyze emotions interactively.

---

## 📚 Libraries Used

- TensorFlow / Keras
- Librosa
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn, Plotly
- Streamlit

---

## 📈 Future Improvements

- Add support for real-time microphone input
- Deploy model as cloud-based API
- Incorporate attention-based models (Transformers)
- Expand to multilingual datasets
- Integrate video emotion recognition for multimodal analysis

---

## 🔖 Author

Devendra Bainda

---

## 📄 Acknowledgements

- **RAVDESS Dataset**: Ryerson University

---

