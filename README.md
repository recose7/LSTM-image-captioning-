# 📷 AI Image Captioning Engine

An end-to-end deep learning proof-of-concept that generates contextually relevant English descriptions for uploaded images. This project combines computer vision and natural language processing, packaged into a lightweight, interactive web application.

## 🧠 System Architecture
This model utilizes an encoder-decoder architecture, designed to be highly efficient and resource-aware:
* **Vision Encoder (ResNet50):** Extracts 49x2048 spatial feature grids from raw images without pooling.
* **Text Decoder (LSTM):** A 256-unit recurrent neural network handling sequence generation and text embedding.
* **Attention Mechanism:** An Additive Attention layer dynamically weights specific image regions based on the current word being predicted.
* **Footprint:** ~5.7M trainable parameters (~22 MB total model size).

## 📊 Dataset & Training
Trained on the **Flickr8k Dataset** to prove pipeline architecture and deployment viability without requiring massive cloud compute budgets.
* **Data Volume:** 8,091 unique images and 40,455 human-written captions.
* **Vocabulary:** Tokenized to 8,831 unique words (max sequence length: 38).
* **Convergence:** Trained for 20 epochs using early stopping and model checkpointing, reaching a final categorical crossentropy loss of ~1.70. 
* **Note on Limitations:** Due to the 8K sample limit and dataset distribution, the model exhibits a known class imbalance bias toward dominant categories (e.g., dogs, outdoor sports).

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)
   cd Your-Repo-Name
