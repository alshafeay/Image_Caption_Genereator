# 🖼️ Image Caption Generator
### *Turn Images into Words with Deep Learning*

![Project Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![Gradio](https://img.shields.io/badge/GUI-Gradio-purple?style=flat-square)

A comprehensive Deep Learning project that combines **Computer Vision** and **Natural Language Processing** to generate descriptive captions for images. This repository contains both the **training pipeline** (Jupyter Notebook) and a **production-ready GUI** (Gradio App).

---

## � Table of Contents
- [Project Overview](#-project-overview)
- [🧠 Part 1: Model Training (The Notebook)](#-part-1-model-training-the-notebook)
    - [Architecture](#architecture)
    - [Dataset](#dataset)
    - [Training Workflow](#training-workflow)
- [✨ Part 2: Detection App (The GUI)](#-part-2-detection-app-the-gui)
    - [Features](#features)
    - [Installation](#installation)
    - [Usage](#usage)
- [📂 Project Structure](#-project-structure)
- [⚖️ License](#-license)

---

## 🚀 Project Overview

This project implements an **Encoder-Decoder** architecture to solve the image captioning problem. The model learns to recognize visual content in an image and translate it into a coherent English sentence.

**Core Technologies:**
*   **Encoder:** `InceptionV3` (Input: Image $\rightarrow$ Output: 2048-dim feature vector)
*   **Decoder:** `LSTM` (Input: Sequence $\rightarrow$ Output: Next word probability)
*   **Fusion:** Feature concatenation method
*   **Interface:** Web-based GUI using `Gradio`

---

## 🧠 Part 1: Model Training (The Notebook)

The file `Image_Caption_Generator.ipynb` contains the complete end-to-end training pipeline, designed to run on **Google Colab**.

### Architecture
We utilize a **Merge Architecture**:
1.  **Image Model (Encoder):**
    *   Takes an input image (299x299).
    *   Uses **InceptionV3** (pre-trained on ImageNet) with the top layer removed.
    *   Extracts a **2048-dimensional** feature vector.
    *   Passes through a dense layer (256 units).
2.  **Caption Model (Decoder):**
    *   Takes partial caption sequences.
    *   Uses an **Embedding Layer** (256 dims) to handle text input.
    *   Uses an **LSTM Layer** (256 units) to learn sequence dependencies.
3.  **Merger:**
    *   Combines the outputs of both models via addition.
    *   Final Dense layer with **Softmax activation** to predict the next word from the vocabulary (8780 unique words).

### Dataset
*   **Source:** [Flickr8k Dataset](https://forms.illinois.edu/sec/1713398)
*   **Size:** 8,092 images
*   **Captions:** 5 captions per image (Total: ~40,000 captions)
*   **Splits:** 6,000 Train | 1,000 Validation | 1,000 Test

### Training Workflow
1.  **Feature Extraction:** Run all images through InceptionV3 and save features to `image_features.pkl` (Optimization to avoid re-processing).
2.  **Text Preprocessing:** Lowercase, remove punctuation/numbers, add `<start>`/`<end>` tokens.
3.  **Tokenization:** Vectorize text to integers.
4.  **Data Generator:** Yields batches of `(image_feature, partial_sequence) -> next_word` for memory efficiency.
5.  **Training:** Optimization using Categorical Crossentropy and Adam optimizer.

---

## ✨ Part 2: Detection App (The GUI)

The `app.py` file launches a stunning, dark-themed web interface to test the model in real-time.

### Features
*   ✅ **Beam Search Decoding:** Selectable beam width (1-5) for higher quality captions compared to greedy search.
*   ✅ **CPU Optimized:** Runs efficiently on local machines without needing a GPU.
*   ✅ **Interactive UI:** Drag-and-drop image upload with instant generation.
*   ✅ **Modern Design:** Glassmorphism aesthetics with smooth animations.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/alshafeay/Image_Caption_Genereator.git
    cd Image_Caption_GUI
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Model Weights:**
    *   Ensure `best_model.keras` and `tokenizer.pkl` are in the directory.
    *   *Note: First run will download InceptionV3 weights (~90MB).*

### Usage

**Run the application:**
```bash
python app.py
```
*   The app will open in your default browser at `http://127.0.0.1:7862`
*   Upload an image, select **Beam Width** (higher = better quality), and click **Generate Caption**.

---

## � Project Structure

```bash
Image_Caption_GUI/
├── Image_Caption_Generator.ipynb  # 📓 Training Notebook (The Brain)
├── app.py                         # 🚀 Gradio Application (The Interface)
├── requirements.txt               # 📦 Dependencies
├── model/
│   ├── best_model.keras           # Trained Model
│   └── image_features.pkl         # (Optional) Pre-computed features
├── examples/                      # 🖼️ Sample images for testing
├── tokenizer.pkl                  # 🔤 Word Tokenizer object
├── model_config.pkl               # ⚙️ Configuration (max_length, vocab_size)
└── README.md                      # 📄 This file
```

---

## ⚖️ License
This project is for educational purposes as part of a Deep Learning course.
*   Dataset: [Flickr8k Terms of Use](https://forms.illinois.edu/sec/1713398)
*   Model: Custom implementation using TensorFlow/Keras.

---
*Created with ❤️ by Ahmed Alshafeay*
