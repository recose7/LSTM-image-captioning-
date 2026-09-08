import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from PIL import Image
import json
import traceback

# --- 1. USER INTERFACE SETUP ---
st.set_page_config(page_title="AI Image Captioning", layout="centered")
st.title("📷 AI Image Captioning Engine")
st.write("Upload an image and the Attention-based LSTM model will describe it!")

status_text = st.empty()
status_text.info("⏳ Loading modern AI assets into memory... (Please wait)")

# --- 2. ASSET LOADING (Modern Standards Only) ---
@st.cache_resource
def load_all_assets():
    # 1. Load the modern .keras model 
    caption_model = tf.keras.models.load_model('attention_caption_model.keras')
    
    # 2. Load ResNet50 Feature Extractor
    resnet_model = ResNet50(weights='imagenet', include_top=False)
    
    # 3. Reconstruct Tokenizer from JSON
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
        tokenizer = tokenizer_from_json(f.read())
        
    # 4. Load Config
    with open('model_config.json', 'r', encoding='utf-8') as f:
        max_length = json.load(f)['max_length']
        
    # Reverse dictionary for translating numbers back to English words
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    
    return caption_model, resnet_model, tokenizer, max_length, index_to_word

# --- 3. EXPLICIT ERROR CATCHING ---
try:
    caption_model, resnet_model, tokenizer, max_length, index_to_word = load_all_assets()
    status_text.success("✅ Models loaded! Ready for inference.")
except Exception as e:
    status_text.error(f"❌ Model Loading Failed: {e}")
    st.code(traceback.format_exc())
    st.stop()

# --- 4. INFERENCE ENGINE ---
def generate_greedy_caption(image, caption_model, resnet_model, max_length, tokenizer, index_to_word):
    # Process the user's uploaded image
    image = image.resize((224, 224))
    image_pixels = tf.keras.preprocessing.image.img_to_array(image)
    image_pixels = np.expand_dims(image_pixels, axis=0)
    image_pixels = preprocess_input(image_pixels)

    # Extract spatial features using ResNet50
    image_feature = resnet_model.predict(image_pixels, verbose=0)
    image_feature = image_feature.reshape(1, 49, 2048)

    start_id = tokenizer.word_index['startseq']
    end_id = tokenizer.word_index['endseq']

    sequence = np.zeros((1, max_length), dtype=np.int32)
    sequence[0, 0] = start_id
    predicted_words = []

    # Autoregressive Loop
    for i in range(max_length - 1):
        # Explicit dictionary mapping to prevent Keras tensor flip crashes
        yhat = caption_model.predict({
            'Image_features': image_feature, 
            'Text_sequence': sequence
        }, verbose=0)
        
        # Get the highest probability word
        next_word_id = np.argmax(yhat[0, i, :])
        
        # Stop if the model predicts the end token
        if next_word_id == end_id:
            break
            
        # Add the predicted word ID to our sequence for the next loop iteration
        sequence[0, i + 1] = next_word_id
        
        # Translate ID to English word
        word = index_to_word.get(next_word_id, '')
        if word:
            predicted_words.append(word)

    return ' '.join(predicted_words)

# --- 5. FRONTEND FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    raw_image = Image.open(uploaded_file).convert('RGB')
    st.image(raw_image, caption='Uploaded Image', use_container_width=True)
    
    # Generate button
    if st.button("Generate Caption"):
        with st.spinner("Analyzing spatial features and generating text..."):
            caption = generate_greedy_caption(
                raw_image, caption_model, resnet_model, max_length, tokenizer, index_to_word
            )
            st.success(f"**Prediction:** {caption}")