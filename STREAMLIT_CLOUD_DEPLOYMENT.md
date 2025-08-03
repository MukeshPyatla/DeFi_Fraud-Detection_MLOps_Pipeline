# 🚀 Streamlit Cloud Deployment Guide

This guide will help you deploy the DeFi Fraud Detection Dashboard on Streamlit Cloud.

## 📋 Prerequisites

1. **GitHub Account**: You need a GitHub account to host your code
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Python Knowledge**: Basic understanding of Python and Streamlit

## 🔧 Step-by-Step Deployment

### 1. Prepare Your Repository

Make sure your repository has the following structure:

```
your-repo/
├── streamlit_app.py          # Main Streamlit app
├── requirements_streamlit.txt # Streamlit-specific requirements
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── README.md                 # Project documentation
└── .gitignore               # Git ignore file
```

### 2. Create a GitHub Repository

```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: DeFi Fraud Detection Dashboard"

# Create GitHub repository and push
git remote add origin https://github.com/yourusername/your-repo-name.git
git branch -M main
git push -u origin main
```

### 3. Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)
2. **Sign in**: Use your GitHub account to sign in
3. **New App**: Click "New app"
4. **Configure Deployment**:
   - **Repository**: Select your GitHub repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Choose a custom URL (optional)

5. **Advanced Settings** (Optional):
   - **Python version**: 3.9
   - **Requirements file**: `requirements_streamlit.txt`

6. **Deploy**: Click "Deploy!"

## ⚙️ Configuration Options

### Environment Variables

You can set environment variables in Streamlit Cloud:

1. Go to your app settings in Streamlit Cloud
2. Navigate to "Secrets"
3. Add your configuration:

```toml
[api]
base_url = "https://your-api-url.com"
api_key = "your-api-key"

[ethereum]
rpc_url = "https://mainnet.infura.io/v3/YOUR_PROJECT_ID"
```

### Custom Domain (Optional)

1. In your app settings, go to "Custom domain"
2. Add your custom domain
3. Update your DNS settings as instructed

## 🔍 Features Available

### Demo Mode (Default)
- ✅ Interactive fraud prediction
- ✅ Real-time transaction monitoring
- ✅ Model performance visualization
- ✅ Audit trail simulation
- ✅ Sample data and charts

### Connected Mode (With API)
- ✅ Full API integration
- ✅ Real blockchain data
- ✅ Live model predictions
- ✅ Actual fraud detection

## 🛠️ Customization

### 1. Modify the Dashboard

Edit `streamlit_app.py` to customize:

```python
# Add new pages
def show_custom_page():
    st.header("Custom Page")
    # Your custom content here

# Add to navigation
page = st.sidebar.selectbox(
    "Choose a page",
    ["Overview", "Custom Page", "Settings"]
)
```

### 2. Add New Features

```python
# Add new visualizations
def show_custom_chart():
    data = pd.DataFrame({
        'x': range(10),
        'y': np.random.randn(10)
    })
    fig = px.line(data, x='x', y='y')
    st.plotly_chart(fig)

# Add new metrics
def show_custom_metrics():
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Custom Metric", "Value", "Delta")
```

### 3. Connect to External APIs

```python
# Add API integration
def get_external_data():
    try:
        response = requests.get("https://api.example.com/data")
        return response.json()
    except:
        return {"error": "API unavailable"}
```

## 🔒 Security Considerations

### 1. API Keys
- Store sensitive data in Streamlit secrets
- Never commit API keys to your repository
- Use environment variables for configuration

### 2. Data Privacy
- The demo version doesn't store any personal data
- All data is processed in memory
- No persistent storage on Streamlit Cloud

### 3. Rate Limiting
- Be mindful of API rate limits
- Implement caching for external API calls
- Use demo data when APIs are unavailable

## 📊 Monitoring Your Deployment

### 1. App Health
- Check your app status in Streamlit Cloud dashboard
- Monitor deployment logs for errors
- Set up alerts for app downtime

### 2. Performance
- Monitor app response times
- Optimize data processing for large datasets
- Use caching for expensive operations

### 3. Usage Analytics
- Track user engagement
- Monitor feature usage
- Analyze user feedback

## 🚨 Troubleshooting

### Common Issues

1. **App Not Loading**
   ```bash
   # Check requirements
   pip install -r requirements_streamlit.txt
   
   # Test locally
   streamlit run streamlit_app.py
   ```

2. **Import Errors**
   ```python
   # Make sure all imports are in requirements_streamlit.txt
   # Check for missing dependencies
   ```

3. **API Connection Issues**
   ```python
   # Add error handling
   try:
       response = requests.get(api_url)
   except requests.exceptions.RequestException:
       st.warning("API unavailable, using demo mode")
   ```

### Debug Mode

Enable debug mode locally:

```bash
streamlit run streamlit_app.py --logger.level debug
```

## 🔄 Continuous Deployment

### 1. Automatic Updates
- Streamlit Cloud automatically redeploys when you push to your main branch
- No manual intervention required

### 2. Version Control
```bash
# Make changes
git add .
git commit -m "Update dashboard features"
git push origin main
```

### 3. Rollback
- Use git tags for version management
- Revert to previous commits if needed
- Test changes locally before pushing

## 📈 Scaling Considerations

### 1. Performance Optimization
- Use caching for expensive operations
- Optimize data processing
- Minimize external API calls

### 2. Memory Management
- Streamlit Cloud has memory limits
- Avoid loading large datasets in memory
- Use streaming for large data processing

### 3. Concurrent Users
- Streamlit Cloud handles multiple users
- Implement proper session management
- Use state management for user data

## 🎯 Best Practices

### 1. Code Organization
- Keep your main app file clean
- Separate concerns into functions
- Use proper error handling

### 2. User Experience
- Provide clear navigation
- Add loading indicators
- Include helpful error messages

### 3. Documentation
- Add comments to your code
- Include usage instructions
- Document API integrations

## 🚀 Advanced Features

### 1. Custom Components
```python
# Create custom Streamlit components
import streamlit.components.v1 as components

def custom_component():
    components.html("""
        <div>Custom HTML Component</div>
    """)
```

### 2. Session State
```python
# Use session state for user data
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

st.session_state.user_data['preference'] = st.selectbox("Preference", ["A", "B"])
```

### 3. File Upload
```python
# Handle file uploads
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
```

## 📞 Support

- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **Streamlit Community**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Report bugs in your repository

---

**Happy Deploying! 🚀✨** 