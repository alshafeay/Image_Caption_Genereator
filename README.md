# Image Caption Generator - Gradio GUI

## 📋 Description
A local Gradio-based GUI application for generating captions for images using a deep learning model trained on the Flickr8k dataset. The model uses CNN (InceptionV3) for image encoding and LSTM for caption generation with beam search decoding.

---

## 🎯 Features
- ✅ **Upload any image** and get AI-generated captions
- ✅ **Beam search decoding** for higher quality captions
- ✅ **Adjustable beam width** (1-5) for quality vs speed tradeoff
- ✅ **CPU-optimized** - No GPU required
- ✅ **Modern web interface** with Gradio
- ✅ **Example images** included for testing
- ✅ **Processing time display** to monitor performance

---

## 📁 Project Structure
```
Image_Caption_GUI/
├── model/
│   ├── best_model.keras          # Trained caption generation model
│   └── image_features.pkl        # Pre-extracted training features (not used in GUI)
├── examples/
│   └── *.jpg                     # Sample test images
├── tokenizer.pkl                 # Word tokenizer
├── model_config.pkl              # Model configuration
├── app.py                        # Main Gradio GUI application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## ⚙️ Installation

### Step 1: Install Python
Make sure you have **Python 3.8 or higher** installed.

Check version:
```bash
python --version
```

### Step 2: Install Dependencies
Navigate to this directory and run:

```bash
pip install -r requirements.txt
```

**Note:** First install will download InceptionV3 weights (~90 MB) from the internet.

---

## 🚀 Usage

### Running the GUI

1. Open terminal/command prompt in this directory
2. Run:
   ```bash
   python app.py
   ```
3. Your browser will automatically open at: `http://127.0.0.1:7860`
4. Upload an image and click "Generate Caption"!

### Beam Width Settings
- **Beam Width 1**: Fastest (greedy search) - ~2 seconds
- **Beam Width 3**: Balanced (recommended) - ~3 seconds
- **Beam Width 5**: Best quality (slower) - ~5 seconds

---

## 💻 System Requirements

### Minimum:
- **RAM**: 4 GB
- **Storage**: 2 GB free space
- **CPU**: Any modern processor (no GPU needed)
- **OS**: Windows, macOS, or Linux

### Recommended:
- **RAM**: 8 GB or more
- **CPU**: Multi-core processor for faster inference

---

## 🧠 Model Details

### Architecture:
- **Encoder**: InceptionV3 (pre-trained on ImageNet)
  - Extracts 2048-dimensional feature vectors
- **Decoder**: LSTM with 256 units
  - Embedding dimension: 256
  - Max caption length: 37 words
- **Decoding**: Beam search with configurable width

### Training:
- **Dataset**: Flickr8k (8,000 images with 40,000 captions)
- **Vocabulary**: 8,780 unique words
- **Training split**: 6,000 images
- **Validation split**: 1,000 images
- **Test split**: 1,000 images

---

## 📊 Performance

### Processing Time (on typical CPU):
- Feature extraction: ~1-2 seconds
- Caption generation (beam=3): ~1-2 seconds
- **Total**: ~2-4 seconds per image

### Quality:
- Uses beam search for better caption quality
- Higher beam width = better captions but slower

---

## 🎨 Example Usage

### Via GUI:
1. Click "Upload Image" or drag & drop
2. Adjust beam width slider (default: 3)
3. Click "Generate Caption"
4. View your caption!

### Sample Outputs:
- Dog playing: *"A brown dog is running through the grass"*
- Beach scene: *"A person is walking on the beach"*
- City street: *"A man is riding a bike on the street"*

---

## 🛠️ Troubleshooting

### Issue: "No module named 'tensorflow'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Slow performance
**Solutions**:
- Reduce beam width to 1 or 2
- Close other applications to free up RAM
- First run is slower (loading models)

### Issue: Port already in use
**Solution**: Change port in app.py:
```python
demo.launch(server_port=7861)  # Change 7860 to 7861
```

---

## 🔧 Advanced Configuration

### Change Port:
Edit `app.py`, line ~380:
```python
demo.launch(server_port=7860)  # Change this number
```

### Disable Auto-Open Browser:
Edit `app.py`, line ~380:
```python
demo.launch(inbrowser=False)
```

### Enable Public Sharing:
Edit `app.py`, line ~380:
```python
demo.launch(share=True)  # Creates public link (temporary)
```

---

## 📚 Technical Stack

- **Framework**: TensorFlow 2.x + Keras
- **GUI**: Gradio 4.x
- **Image Processing**: Pillow, NumPy
- **Pre-trained Model**: InceptionV3
- **Language**: Python 3.8+

---

## 📝 Notes

- The GUI runs entirely on your local machine (no data sent to cloud)
- First run downloads InceptionV3 weights (~90 MB)
- CPU inference is slower than GPU but works on any computer
- Model trained on general images (Flickr8k dataset)
- Best results with natural, well-lit photos

---

## 🎓 Academic Use

This project was created as part of a Deep Learning course. The model demonstrates:
- Transfer learning with InceptionV3
- Sequence-to-sequence modeling with LSTM
- Beam search decoding for text generation
- End-to-end deployment with Gradio

---

## 📄 License

For academic/educational purposes.

---

## 🤝 Support

If you encounter any issues:
1. Check the Troubleshooting section above
2. Verify all files are in correct locations
3. Ensure Python version is 3.8+
4. Try reinstalling dependencies

---

**Enjoy generating captions! 🎉**
