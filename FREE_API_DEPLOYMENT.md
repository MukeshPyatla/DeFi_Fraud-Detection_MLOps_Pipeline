# 🚀 Free API Deployment Guide

This guide shows you how to deploy the DeFi Fraud Detection API for free on various platforms.

## 📋 Prerequisites

1. **GitHub Account**: Your code is already on GitHub
2. **Free Platform Account**: Choose one of the platforms below

## 🎯 Option 1: Render (Recommended)

**Render** offers a generous free tier with 750 hours/month.

### Step 1: Sign Up
1. Go to [render.com](https://render.com)
2. Sign up with your GitHub account

### Step 2: Create New Web Service
1. Click "New +" → "Web Service"
2. Connect your GitHub repository: `MukeshPyatla/DeFi_Fraud-Detection_MLOps_Pipeline`

### Step 3: Configure Service
```
Name: defi-fraud-api
Environment: Python 3
Build Command: pip install -r requirements_api.txt
Start Command: uvicorn api_app:app --host 0.0.0.0 --port $PORT
```

### Step 4: Deploy
1. Click "Create Web Service"
2. Wait for deployment (2-3 minutes)
3. Your API will be available at: `https://your-app-name.onrender.com`

## 🎯 Option 2: Railway

**Railway** has a simple deployment process.

### Step 1: Sign Up
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub

### Step 2: Deploy
1. Click "New Project" → "Deploy from GitHub repo"
2. Select your repository
3. Railway will auto-detect it's a Python app
4. Set the start command: `uvicorn api_app:app --host 0.0.0.0 --port $PORT`

## 🎯 Option 3: Heroku

**Heroku** has a free tier (with limitations).

### Step 1: Install Heroku CLI
```bash
# macOS
brew install heroku/brew/heroku

# Or download from https://devcenter.heroku.com/articles/heroku-cli
```

### Step 2: Login and Deploy
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-defi-fraud-api

# Add Python buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

## 🎯 Option 4: Local Development

For testing locally:

### Step 1: Install Dependencies
```bash
pip install -r requirements_api.txt
```

### Step 2: Run API
```bash
python api_app.py
```

The API will be available at: `http://localhost:8000`

## 🔧 Configuration

### Environment Variables (Optional)
Set these in your deployment platform:

```
PORT=8000
ENVIRONMENT=production
```

### API Endpoints

Once deployed, your API will have these endpoints:

- **Health Check**: `GET /api/v1/health`
- **Model Info**: `GET /api/v1/model/info`
- **Predict Fraud**: `POST /api/v1/predict`
- **Metrics**: `GET /api/v1/metrics`

## 🔗 Connect to Dashboard

After deploying your API:

1. **Get your API URL** (e.g., `https://your-app.onrender.com`)
2. **Update Streamlit Cloud**:
   - Go to your Streamlit app settings
   - Add environment variable: `API_BASE_URL=https://your-app.onrender.com`
3. **Redeploy** your Streamlit app

## 📊 Testing Your API

### Test with curl:
```bash
# Health check
curl https://your-app.onrender.com/api/v1/health

# Predict fraud
curl -X POST https://your-app.onrender.com/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_hash": "0x1234567890abcdef",
    "from_address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
    "to_address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
    "value_eth": 1.5,
    "gas_price": 20,
    "gas_used": 21000,
    "block_number": 18000000
  }'
```

## 🎉 Success!

Once deployed, your dashboard will show:
- ✅ **API Status**: Connected
- ✅ **Real-time predictions**
- ✅ **Model performance data**
- ✅ **Live fraud detection**

## 🆘 Troubleshooting

### Common Issues:

1. **Build fails**: Check that `requirements_api.txt` exists
2. **Port issues**: Make sure to use `$PORT` environment variable
3. **CORS errors**: The API includes CORS middleware for dashboard access
4. **Timeout**: Free tiers may have cold starts

### Support:
- Check platform-specific logs
- Verify all files are committed to GitHub
- Test locally first with `python api_app.py`

---

**🎯 Recommended**: Start with **Render** - it's the easiest and most reliable free option! 