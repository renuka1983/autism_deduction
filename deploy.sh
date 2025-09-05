#!/bin/bash

# Autism Detection App - Deployment Script
echo "🚀 Deploying Autism Detection App to Streamlit Cloud..."

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Xcode Command Line Tools first."
    echo "Run: xcode-select --install"
    exit 1
fi

# Initialize git repository if not already done
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Add all files
echo "📦 Adding files to git..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit."
else
    echo "💾 Committing changes..."
    git commit -m "Deploy: Autism Detection App for Streamlit Cloud"
fi

# Check if remote origin exists
if ! git remote get-url origin &> /dev/null; then
    echo "⚠️  No remote repository configured."
    echo "Please create a GitHub repository and run:"
    echo "git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
    echo "git branch -M main"
    echo "git push -u origin main"
    echo ""
    echo "Then deploy at: https://share.streamlit.io"
else
    echo "🚀 Pushing to GitHub..."
    git branch -M main
    git push -u origin main
    echo "✅ Code pushed to GitHub!"
    echo "Now go to https://share.streamlit.io to deploy your app!"
fi

echo ""
echo "📋 Next steps:"
echo "1. Go to https://share.streamlit.io"
echo "2. Sign in with GitHub"
echo "3. Click 'New app'"
echo "4. Select your repository"
echo "5. Set main file to 'app.py'"
echo "6. Deploy!"
echo ""
echo "🎉 Your app will be available at: https://YOUR_APP_NAME.streamlit.app"
