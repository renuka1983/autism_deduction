# 🔧 Troubleshooting Guide

## Common Streamlit Cloud Deployment Issues

### 1. Requirements Installation Error

**Problem**: "Error installing requirements" when deploying to Streamlit Cloud.

**Solutions**:

#### Option A: Use Minimal Requirements
Replace your `requirements.txt` with the minimal version:
```bash
# Rename the current requirements.txt
mv requirements.txt requirements-full.txt
# Use the minimal version
mv requirements-minimal.txt requirements.txt
```

#### Option B: Check Streamlit Cloud Logs
1. Go to your app in Streamlit Cloud
2. Click "Manage App"
3. Check the "Logs" tab for specific error messages
4. Look for dependency conflicts or memory issues

#### Option C: Use Compatible Versions
If you still get errors, try this ultra-minimal requirements.txt:
```
streamlit
opencv-python-headless
mediapipe
librosa
matplotlib
scikit-learn
tensorflow-cpu
reportlab
numpy
Pillow
```

### 2. Memory Issues

**Problem**: App crashes due to memory limits.

**Solutions**:
- Use `tensorflow-cpu` instead of `tensorflow`
- Use `opencv-python-headless` instead of `opencv-python`
- Remove unnecessary dependencies

### 3. Import Errors

**Problem**: Module not found errors.

**Solutions**:
- Check that all files are in the correct directory structure
- Ensure `utils/` directory contains all required files
- Verify import paths are correct

### 4. Model Loading Issues

**Problem**: Models not loading.

**Solutions**:
- Ensure model files are in the `models/` directory
- Check file permissions
- Verify model file formats (.h5 for TensorFlow)

### 5. File Upload Issues

**Problem**: File uploads not working.

**Solutions**:
- Check file size limits (Streamlit Cloud has limits)
- Ensure file types are supported
- Verify temporary file handling

## Debugging Steps

### 1. Check App Logs
```bash
# In Streamlit Cloud dashboard
1. Go to your app
2. Click "Manage App"
3. Check "Logs" tab
4. Look for error messages
```

### 2. Test Locally First
```bash
# Test the app locally before deploying
streamlit run app.py
```

### 3. Check Dependencies
```bash
# Test requirements installation
pip install -r requirements.txt
```

### 4. Verify File Structure
```
your-repo/
├── app.py                    # Main app
├── requirements.txt          # Dependencies
├── utils/                    # Utility modules
│   ├── model_loader.py
│   ├── preprocess.py
│   └── pairs_config.py
└── models/                   # Model files (optional)
```

## Quick Fixes

### For Requirements Error:
1. Use `requirements-minimal.txt`
2. Remove version constraints
3. Use CPU-only versions of heavy libraries

### For Memory Error:
1. Use `tensorflow-cpu`
2. Use `opencv-python-headless`
3. Reduce model sizes

### For Import Error:
1. Check file structure
2. Verify all files are committed
3. Check import paths

## Getting Help

1. **Streamlit Cloud Logs**: Check your app's logs first
2. **Streamlit Forums**: https://discuss.streamlit.io
3. **GitHub Issues**: Create an issue in your repository
4. **Streamlit Docs**: https://docs.streamlit.io/streamlit-community-cloud

## Emergency Fallback

If nothing works, try this ultra-minimal app:

```python
import streamlit as st

st.title("Autism Detection App")
st.write("App is loading...")

# Add your functionality here
```

Then gradually add dependencies back.
