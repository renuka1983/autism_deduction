#!/bin/bash

echo "🚀 Pushing Autism Detection App to GitHub..."
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Please run 'git init' first."
    exit 1
fi

# Check if there are uncommitted changes
if ! git diff --staged --quiet; then
    echo "📝 Committing changes..."
    git add .
    git commit -m "Update: Enhanced autism detection app for deployment"
fi

# Check if remote exists
if ! git remote get-url origin &> /dev/null; then
    echo "⚠️  No GitHub repository configured yet."
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://github.com/new"
    echo "2. Create a new repository named 'autism-detector-app'"
    echo "3. Don't initialize with README (we already have files)"
    echo "4. Copy the repository URL"
    echo "5. Run this command with your repository URL:"
    echo "   git remote add origin YOUR_REPOSITORY_URL"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    echo "Or run: ./push_to_github.sh YOUR_REPOSITORY_URL"
    exit 0
fi

# Push to GitHub
echo "📤 Pushing to GitHub..."
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo "✅ Successfully pushed to GitHub!"
    echo ""
    echo "🎉 Next steps:"
    echo "1. Go to https://share.streamlit.io"
    echo "2. Sign in with GitHub"
    echo "3. Click 'New app'"
    echo "4. Select your repository"
    echo "5. Set main file to 'app.py'"
    echo "6. Deploy!"
else
    echo "❌ Failed to push to GitHub. Please check your repository URL and try again."
fi
