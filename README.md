# Autism Detection App

A comprehensive multi-modal autism screening application that analyzes facial landmarks, handwriting patterns, and voice characteristics to provide an ensemble prediction for autism spectrum disorder (ASD) screening.

## Features

- **Multi-Modal Analysis**: Combines three different data types for comprehensive screening
  - **Facial Landmarks**: Uses MediaPipe to extract facial feature distances
  - **Handwriting Analysis**: CNN-based analysis of handwriting patterns
  - **Voice Analysis**: MFCC feature extraction with LSTM for voice characteristics
- **Real-time Processing**: Upload and analyze data instantly
- **Visual Feedback**: Annotated facial landmarks and handwriting previews
- **PDF Reports**: Generate comprehensive reports with results and visualizations
- **Training Dashboard**: Built-in model training interface

## Architecture

### Models
- **Face Model**: Dense neural network analyzing facial landmark distances
- **Handwriting Model**: CNN processing preprocessed handwriting images (128x128)
- **Audio Model**: LSTM processing MFCC features from voice recordings

### Data Processing
- **Face**: MediaPipe facial mesh → landmark extraction → distance calculations
- **Handwriting**: Image preprocessing → grayscale conversion → resizing
- **Audio**: Librosa MFCC extraction → padding/truncation → LSTM processing

## Installation

### Local Development

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd autism_detector_app_final
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. **Prepare your repository**:
   - Ensure all files are in the root directory
   - Models should be in the `models/` directory
   - Data should be in the `data/` directory (if training)

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set the main file path to `app.py`
   - Deploy!

## Usage

### For Screening
1. Upload a **face image** (required)
2. Optionally upload **handwriting image** and/or **voice recording**
3. Enter child ID (optional)
4. Click "Predict" to get results
5. Download PDF report if desired

### For Training
1. Run `streamlit run train_dashboard.py`
2. Ensure training data is in the `data/` directory structure:
   ```
   data/
   ├── face/
   │   ├── autistic/
   │   └── non_autistic/
   ├── handwriting/
   │   ├── autistic/
   │   └── non_autistic/
   └── voice/
       ├── autistic/
       └── non_autistic/
   ```
3. Click "Train All" to train all three models

## File Structure

```
autism_detector_app_final/
├── app.py                    # Main Streamlit application
├── train_dashboard.py        # Training interface
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── utils/
│   ├── model_loader.py      # Model loading utilities
│   ├── preprocess.py        # Data preprocessing functions
│   └── pairs_config.py      # Facial landmark pairs configuration
├── models/                  # Trained model files (generated)
│   ├── face_model.h5
│   ├── handwriting_model.h5
│   └── audio_model.h5
└── data/                    # Training data (if available)
    ├── face/
    ├── handwriting/
    └── voice/
```

## Configuration

### Facial Landmark Pairs
The app uses specific MediaPipe facial landmark pairs for distance calculations. These are defined in `utils/pairs_config.py` and can be modified based on your requirements.

### Model Parameters
- **Face Model**: 128 → 64 → 1 (Dense layers)
- **Handwriting Model**: Conv2D → MaxPool → Conv2D → MaxPool → Dense
- **Audio Model**: TimeDistributed Conv2D → LSTM → Dense

## Requirements

- Python 3.8+
- Streamlit 1.28.0+
- TensorFlow 2.13.0+
- OpenCV 4.8.0+
- MediaPipe 0.10.0+
- Librosa 0.10.0+
- And other dependencies listed in `requirements.txt`

## Important Notes

⚠️ **Medical Disclaimer**: This application is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis or screening.

⚠️ **Data Privacy**: Ensure all uploaded data is handled according to privacy regulations and best practices.

## Troubleshooting

### Common Issues

1. **Models not loading**: Ensure model files exist in the `models/` directory
2. **Import errors**: Check that all utility files are in the correct directory structure
3. **Memory issues**: Reduce image sizes or use smaller batch sizes for training
4. **Audio processing errors**: Ensure audio files are in WAV format

### Performance Tips

- Use high-quality images for better facial landmark detection
- Ensure good lighting in face images
- Use clear handwriting samples
- Record audio in quiet environments

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with local regulations regarding medical data and privacy.

## Support

For issues and questions, please create an issue in the repository or contact the development team.
