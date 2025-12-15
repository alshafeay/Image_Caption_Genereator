"""
Image Caption Generator - Gradio GUI
CPU-Optimized with Beam Search

This application generates captions for uploaded images using a trained
CNN-LSTM model with beam search decoding.
""" 

import os
import pickle
import numpy as np
from PIL import Image
import gradio as gr

# TensorFlow/Keras imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow warnings
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Loading models and configuration...")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer.pkl')
CONFIG_PATH = os.path.join(BASE_DIR, 'model_config.pkl')
EXAMPLES_DIR = os.path.join(BASE_DIR, 'examples')

# ============================================================================
# LOAD MODELS AND TOKENIZER
# ============================================================================

# Load configuration
with open(CONFIG_PATH, 'rb') as f:
    config = pickle.load(f)

max_length = config['max_length']
vocab_size = config['vocab_size']

print(f"  Max caption length: {max_length}")
print(f"  Vocabulary size: {vocab_size}")

# Load tokenizer
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

print(f"  Tokenizer loaded with {len(tokenizer.word_index)} words")

# Load caption generation model
caption_model = load_model(MODEL_PATH)
print(f"  Caption model loaded from {MODEL_PATH}")

# Load InceptionV3 for feature extraction
base_model = InceptionV3(weights='imagenet')
feature_extractor = Model(inputs=base_model.input,
                        outputs=base_model.layers[-2].output)
print("  InceptionV3 feature extractor loaded")

print("\n[OK] All models loaded successfully!\n")

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image):
    """
    Preprocess image for InceptionV3 feature extraction
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        features: 2048-dimensional feature vector
    """
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # Resize to InceptionV3 input size
    image = image.resize((299, 299))
    
    # Convert to array
    img_array = img_to_array(image)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess for InceptionV3
    img_array = preprocess_input(img_array)
    
    # Extract features
    features = feature_extractor.predict(img_array, verbose=0)
    
    # Reshape to (2048,)
    features = features.reshape(2048,)
    
    return features

# ============================================================================
# BEAM SEARCH CAPTION GENERATION
# ============================================================================

def beam_search_predictions(model, image_features, tokenizer, max_length, beam_width=3):
    """
    Beam search with improvements to reduce repetition and improve scoring.
    Uses log-probabilities and length normalization, and prevents immediate
    duplicate tokens when extending beams.
    """
    # Resolve start token (support common variants)
    start_candidates = ['start', 'startseq', 'start_token', '<start>', '<startseq>']
    start_token_idx = None
    for name in start_candidates:
        if name in tokenizer.word_index:
            start_token_idx = tokenizer.word_index[name]
            break
    if start_token_idx is None:
        # fallback to 1 if present, otherwise first token index
        start_token_idx = tokenizer.word_index.get('start', None) or next(iter(tokenizer.word_index.values()))

    end_candidates = set(['end', 'endseq', 'end_token', '<end>', '<endseq>'])

    # Prepare image features
    img_feat = np.asarray(image_features)
    if img_feat.ndim == 1:
        img_feat = img_feat.reshape(1, -1)

    # Beam: list of tuples (sequence, logprob_sum)
    beams = [([start_token_idx], 0.0)]
    alpha = 0.7  # length normalization hyperparameter

    while True:
        all_candidates = []

        for seq, logprob in beams:
            # If last token is an end token, keep beam as-is
            last_word = tokenizer.index_word.get(seq[-1])
            if last_word in end_candidates:
                all_candidates.append((seq, logprob))
                continue

            # Pad sequence
            padded = pad_sequences([seq], maxlen=max_length, padding='post')

            # Predict next-word probabilities
            preds = model.predict([img_feat, padded], verbose=0)
            probs = preds[0]
            log_probs = np.log(np.maximum(probs, 1e-12))

            # Top candidates for this beam
            top_indices = np.argsort(log_probs)[-beam_width:][::-1]

            for idx in top_indices:
                # avoid immediate duplicate token
                if len(seq) > 0 and seq[-1] == int(idx):
                    continue
                new_seq = seq + [int(idx)]
                new_logprob = logprob + float(log_probs[idx])
                all_candidates.append((new_seq, new_logprob))

        if not all_candidates:
            break

        # Sort candidates by length-normalized score
        beams = sorted(
            all_candidates,
            key=lambda x: x[1] / (len(x[0]) ** alpha),
            reverse=True
        )[:beam_width]

        # Stopping criteria: all beams ended or max length reached
        ended = True
        for seq, _ in beams:
            if len(seq) < max_length:
                last_word = tokenizer.index_word.get(seq[-1])
                if last_word not in end_candidates:
                    ended = False
                    break
        if ended:
            break

        if all(len(seq) >= max_length for seq, _ in beams):
            break

    # Choose best beam by normalized score
    best = max(beams, key=lambda x: x[1] / (len(x[0]) ** alpha))
    best_sequence = best[0]

    # Convert to words, remove start/end
    final_caption = []
    for idx in best_sequence:
        word = tokenizer.index_word.get(idx)
        if not word or word in ['start', 'end']:
            continue
        final_caption.append(word)

    return ' '.join(final_caption)

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def generate_caption(image, beam_width):
    """
    Main function for Gradio interface
    
    Args:
        image: Uploaded image (PIL Image or numpy array)
        beam_width: Beam search width (1-5)
    
    Returns:
        caption: Generated caption string
    """
    try:
        # Extract features from image
        print("Extracting image features...")
        features = preprocess_image(image)
        
        # Generate caption using beam search
        print(f"Generating caption with beam width={beam_width}...")
        caption = beam_search_predictions(
            caption_model, 
            features, 
            tokenizer, 
            max_length, 
            beam_width=int(beam_width)
        )
        
        # Format output
        caption = caption.capitalize()
        print(f"[OK] Caption generated: {caption}")
        
        return caption
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        return error_msg

# ============================================================================
# PREPARE EXAMPLE IMAGES
# ============================================================================

# Get example images from examples directory
example_images = []
if os.path.exists(EXAMPLES_DIR):
    example_files = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    # Take first 5 examples
    example_images = [os.path.join(EXAMPLES_DIR, f) for f in example_files[:5]]

# ============================================================================
# CREATE GRADIO INTERFACE
# ============================================================================

# Custom CSS for COMPLETE DARK THEME
custom_css = """
/* Hide footer */
footer {visibility: hidden}

/* DARK Background - Black gradient */
.gradio-container {
    max-width: 100% !important;
    width: 100vw !important;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%) !important;
    font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Main content area - DARK */
.contain {
    background: #0f0f0f !important;
    border-radius: 0 !important;
    padding: 40px 60px !important;
    box-shadow: none !important;
    max-width: 100% !important;
    width: 100% !important;
}

/* All blocks dark */
.block {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
}

/* Title styling - Light text */
h1 {
    background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3em !important;
    font-weight: 800 !important;
    text-align: center !important;
    margin-bottom: 10px !important;
    letter-spacing: -1px !important;
}

/* Subtitle styling - Light gray text */
.markdown p {
    text-align: center !important;
    color: #9ca3af !important;
    font-size: 1.1em !important;
    margin-bottom: 30px !important;
}

/* All markdown elements light */
.markdown {
    color: #d1d5db !important;
}

/* Image upload area - Dark */
.image-container, .upload-container {
    background: #1a1a1a !important;
    border-radius: 20px !important;
    border: 3px dashed #8b5cf6 !important;
    transition: all 0.3s ease !important;
}

.image-container:hover, .upload-container:hover {
    border-color: #a855f7 !important;
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(139, 92, 246, 0.3) !important;
}

/* Image component dark */
div[data-testid="image"] {
    background: #1a1a1a !important;
}

/* Generate button styling */
button.primary {
    background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%) !important;
    border: none !important;
    border-radius: 15px !important;
    padding: 18px 40px !important;
    font-size: 1.2em !important;
    font-weight: 700 !important;
    color: white !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    width: 100% !important;
}

button.primary:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 25px rgba(139, 92, 246, 0.6) !important;
}

button.primary:active {
    transform: translateY(-1px);
}

/* Caption output - Dark with light text */
#caption-output {
    font-size: 1.8em !important;
    font-weight: 600 !important;
    color: #f3f4f6 !important;
    background: #1a1a1a !important;
    padding: 40px !important;
    text-align: center !important;
    border-radius: 20px !important;
    border: 2px solid #8b5cf6 !important;
    box-shadow: 0 0 30px rgba(139, 92, 246, 0.2) !important;
    min-height: 200px !important;
    line-height: 1.6 !important;
    font-style: italic !important;
}

#caption-output::before {
    content: '"';
    font-size: 1.5em;
    color: #8b5cf6;
    opacity: 0.5;
}

#caption-output::after {
    content: '"';
    font-size: 1.5em;
    color: #a855f7;
    opacity: 0.5;
}

/* Textbox dark */
textarea, input[type="text"] {
    background: #1a1a1a !important;
    color: #f3f4f6 !important;
    border: 1px solid #2a2a2a !important;
}

/* Label styling - Light */
label {
    font-weight: 600 !important;
    color: #d1d5db !important;
    font-size: 1.1em !important;
    margin-bottom: 10px !important;
}

/* Slider styling - Dark */
input[type="range"] {
    width: 100% !important;
}

.slider {
    background: #1a1a1a !important;
}

/* Info text light */
.info {
    color: #9ca3af !important;
}

/* Example images */
.examples {
    margin-top: 40px !important;
}

.example-image {
    border-radius: 15px !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
    border: 2px solid #2a2a2a !important;
}

.example-image:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 20px rgba(139, 92, 246, 0.4) !important;
    border-color: #8b5cf6 !important;
}

/* Loading animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

/* All panels dark */
.panel {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
}

/* Responsive columns */
.row {
    width: 100% !important;
    gap: 30px !important;
}

.column {
    flex: 1 !important;
    min-width: 400px !important;
}

/* All backgrounds dark */
* {
    scrollbar-color: #4a4a4a #1a1a1a !important;
}

@media (max-width: 1200px) {
    .contain {
        padding: 30px 40px !important;
    }
}

@media (max-width: 768px) {
    .contain {
        padding: 20px !important;
    }
    h1 {
        font-size: 2em !important;
    }
}
"""

# Create Gradio interface - Full-width and visually stunning
with gr.Blocks(title="✨ AI Caption Generator") as demo:
    
    # Header
    gr.Markdown(
        """
        # ✨ AI Caption Generator
        Upload any image and watch AI describe it in words
        """
    )
    
    # Main layout - full width
    with gr.Row():
        with gr.Column(scale=1):
            # Image upload area
            image_input = gr.Image(
                label="📸 Upload Your Image",
                type="pil",
                height=500,
                elem_classes="image-container"
            )
            
            # Beam width slider
            beam_width_slider = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="🔍 Beam Search Width",
                info="Higher = Better quality (1=Fastest, 3=Balanced, 5=Best Quality)"
            )
            
            # Generate button
            generate_btn = gr.Button(
                "✨ Generate Caption",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1):
            # Caption output area - large and beautiful
            caption_output = gr.Textbox(
                label="🎯 AI-Generated Caption",
                lines=6,
                elem_id="caption-output",
                placeholder="Your beautiful caption will appear here...",
                interactive=False
            )
    
    # Examples section (if available)
    if example_images:
        gr.Markdown(
            """
            ---
            ### 🖼️ Try These Examples
            """
        )
        gr.Examples(
            examples=[[img, 3] for img in example_images],
            inputs=[image_input, beam_width_slider],
            outputs=[caption_output],
            fn=generate_caption,
            cache_examples=False
        )
    
    # Connect button to function
    generate_btn.click(
        fn=generate_caption,
        inputs=[image_input, beam_width_slider],
        outputs=[caption_output]
    )

# ============================================================================
# LAUNCH APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI Caption Generator - Starting GUI...")
    print("="*60 + "\n")

    # Try to use preferred port (7862). If it's in use, find a free ephemeral port.
    import socket

    def find_free_port(preferred_port=7862, host='127.0.0.1'):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, preferred_port))
            s.close()
            return preferred_port
        except OSError:
            # Preferred port not available — bind to port 0 to get an ephemeral port
            s.bind((host, 0))
            port = s.getsockname()[1]
            s.close()
            return port

    selected_port = find_free_port(7862)
    print(f"Using port: {selected_port} (if you want a specific port set GRADIO_SERVER_PORT or pass server_port to demo.launch)")

    demo.launch(
        server_name="127.0.0.1",
        server_port=selected_port,
        share=False,
        show_error=True,
        inbrowser=True,
        css=custom_css,
        theme=gr.themes.Glass(primary_hue="purple", secondary_hue="violet").set(
            body_background_fill="*neutral_950",
            body_background_fill_dark="*neutral_950"
        )
    )
