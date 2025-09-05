# Streamlit Cloud Deployment Guide

## Quick Deployment Steps

### 1. Prepare Your Repository
- Ensure all files are committed to your GitHub repository
- The main app file should be `app.py` in the root directory
- All utility files should be in the `utils/` directory

### 2. Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with your GitHub account**
3. **Click "New app"**
4. **Fill in the details**:
   - **Repository**: Select your GitHub repository
   - **Branch**: `main` or `master`
   - **Main file path**: `app.py`
   - **App URL**: Choose a unique URL (e.g., `your-app-name`)
5. **Click "Deploy!"**

### 3. Configure Environment (Optional)

If you need to set environment variables or secrets:

1. Go to your app's settings in Streamlit Cloud
2. Add any required secrets in the "Secrets" section
3. Restart your app

## File Structure for Deployment

```
your-repo/
├── app.py                    # Main application (required)
├── train_dashboard.py        # Training interface (optional)
├── requirements.txt          # Dependencies (required)
├── .streamlit/
│   └── config.toml          # Streamlit config (optional)
├── utils/                   # Utility modules (required)
│   ├── model_loader.py
│   ├── preprocess.py
│   └── pairs_config.py
├── models/                  # Trained models (optional)
│   ├── face_model.h5
│   ├── handwriting_model.h5
│   └── audio_model.h5
├── README.md                # Documentation (recommended)
└── .gitignore              # Git ignore file (recommended)
```

## Important Notes

### Model Files
- If your model files are large (>100MB), consider using Git LFS or hosting them externally
- Models are optional - the app will work without them (showing warnings)
- You can train models using the training dashboard after deployment

### Data Privacy
- Never commit sensitive data to your repository
- Use Streamlit's secrets management for API keys or sensitive configuration
- Ensure compliance with data protection regulations

### Performance Considerations
- Streamlit Cloud has resource limits
- Large model files may cause slower startup times
- Consider optimizing model sizes for better performance

## Troubleshooting

### Common Issues

1. **App won't start**:
   - Check that `app.py` is in the root directory
   - Verify all imports are correct
   - Check the logs in Streamlit Cloud dashboard

2. **Models not loading**:
   - Ensure model files are in the `models/` directory
   - Check file permissions and paths
   - Verify model file formats (.h5 for TensorFlow)

3. **Import errors**:
   - Check that all utility files are in the `utils/` directory
   - Verify Python path and import statements
   - Check that all dependencies are in `requirements.txt`

4. **Memory issues**:
   - Reduce model sizes if possible
   - Optimize image processing
   - Consider using smaller batch sizes

### Getting Help

- Check the [Streamlit Cloud documentation](https://docs.streamlit.io/streamlit-community-cloud)
- Review your app logs in the Streamlit Cloud dashboard
- Create an issue in your repository for specific problems

## Custom Domain (Optional)

If you have a custom domain:

1. Go to your app settings in Streamlit Cloud
2. Add your custom domain
3. Update your DNS settings as instructed
4. Wait for SSL certificate provisioning

## Monitoring and Analytics

Streamlit Cloud provides:
- App usage analytics
- Performance metrics
- Error logs
- User feedback

Access these through your app dashboard in Streamlit Cloud.

## Security Best Practices

1. **Never commit secrets** to your repository
2. **Use environment variables** for sensitive configuration
3. **Regularly update dependencies** for security patches
4. **Monitor app usage** for unusual activity
5. **Implement proper error handling** to avoid information leakage

## Scaling Considerations

For high-traffic applications:
- Consider upgrading to Streamlit Cloud Pro
- Implement caching for expensive operations
- Use external databases for data persistence
- Consider load balancing for multiple instances

---

**Ready to deploy?** Follow the steps above and your autism detection app will be live on Streamlit Cloud!
