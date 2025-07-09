# Streamlit Cloud Deployment Guide

## Prerequisites
1. GitHub repository with your ChanceRAG code
2. OpenAI API key
3. Streamlit Cloud account (https://share.streamlit.io)

## Files Required for Deployment

### Essential Files:
- `app.py` - Main application
- `requirements.txt` - Python dependencies (updated for Python 3.13 compatibility)
- `.python-version` - Python version specification (3.11.9)
- `packages.txt` - System dependencies
- `input.pdf` - Your PDF document

### Configuration Files:
- `.streamlit/config.toml` - Streamlit configuration
- `secrets.toml.example` - Secrets template

## Deployment Steps

### 1. GitHub Setup
```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit for Streamlit deployment"

# Add remote repository
git remote add origin https://github.com/yourusername/chancerag.git

# Push to GitHub
git push -u origin main
```

### 2. Streamlit Cloud Deployment
1. Go to https://share.streamlit.io
2. Click "New app"
3. Connect your GitHub account
4. Select your repository: `yourusername/chancerag`
5. Set main file path: `app.py`
6. Click "Deploy!"

### 3. Configure Secrets
1. In your Streamlit Cloud app dashboard, click "⚙️ Settings"
2. Go to "Secrets" tab
3. Add your secrets:
```toml
OPENAI_API_KEY = "sk-your-actual-api-key-here"
```
4. Click "Save"

### 4. Upload Your PDF
Since Streamlit Cloud apps are stateless, you have a few options:

**Option A: Include PDF in Repository (Recommended for small files)**
- Add your `input.pdf` to the repository
- Commit and push to GitHub
- The app will redeploy automatically

**Option B: Use Streamlit File Uploader**
- Modify `app.py` to include a file uploader
- Users can upload their PDF files dynamically

## Common Issues and Solutions

### Issue: Dependencies failing to install (numpy/distutils error)
**Solution**: Updated requirements use numpy>=1.26.0 which is compatible with Python 3.13 (no distutils dependency)

### Issue: Dependencies failing to install (general)
**Solution**: Use the provided `requirements.txt` with compatible versions

### Issue: Python version compatibility
**Solution**: The `.python-version` file specifies Python 3.11.9 for maximum compatibility

### Issue: API key not found
**Solution**: Ensure OPENAI_API_KEY is set in Streamlit Cloud secrets

### Issue: PDF not found
**Solution**: Make sure `input.pdf` is in your repository root

## Monitoring Deployment
- Check deployment logs in Streamlit Cloud dashboard
- Monitor app performance and errors
- Update requirements as needed

## Local Testing Before Deployment
```bash
# Test with compatible versions
pip install -r requirements.txt
streamlit run app.py
```

## Additional Tips
1. Keep your repository public for easier deployment
2. Use `.gitignore` to exclude sensitive files
3. Test locally before deploying
4. Monitor usage to stay within API limits
5. Consider caching for better performance

## Support
If you encounter issues:
1. Check Streamlit Cloud logs
2. Verify all required files are in repository
3. Ensure secrets are properly configured
4. Test locally with same requirements
