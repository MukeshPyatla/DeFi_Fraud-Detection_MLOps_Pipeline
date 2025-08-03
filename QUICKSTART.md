# 🚀 Quick Start Guide - DeFi Fraud Detection MLOps Pipeline

This guide will help you get the DeFi Fraud Detection Pipeline up and running in minutes!

## 📋 Prerequisites

- Python 3.8+
- pip
- Git
- Docker (optional, for containerized deployment)

## ⚡ Quick Setup

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd DeFi_Fraud_Detection_MLOps_Pipeline

# Run the setup script
python setup.py
```

### 2. Configure Environment

Edit the `.env` file with your settings:

```bash
# Copy environment template
cp env.example .env

# Edit with your configuration
nano .env
```

**Required Configuration:**
```env
# Ethereum RPC URL (get from Infura, Alchemy, etc.)
ETHEREUM_RPC_URL=https://mainnet.infura.io/v3/YOUR_PROJECT_ID

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Dashboard Settings  
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8501
```

### 3. Start the Pipeline

#### Option A: Individual Services

```bash
# Terminal 1: Start the API
python src/api/main.py

# Terminal 2: Start the dashboard
streamlit run src/dashboard/app.py

# Terminal 3: Run data pipeline
python src/data_pipeline/main.py
```

#### Option B: Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### 4. Access the Services

- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **Health Check**: http://localhost:8000/health

## 🔧 First Run

### 1. Train a Model

```bash
# Using the API
curl -X POST "http://localhost:8000/api/v1/model/train" \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "data/processed/train_features.csv",
    "algorithm": "random_forest",
    "test_size": 0.2
  }'
```

### 2. Make Predictions

```bash
# Single prediction
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
    "from_address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
    "to_address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
    "value": 0.1,
    "gas_price": 20.5,
    "gas_used": 21000,
    "block_number": 15000000,
    "timestamp": 1640995200
  }'
```

## 📊 Dashboard Features

### Overview Page
- Real-time transaction monitoring
- Fraud detection metrics
- Model performance indicators

### Real-time Monitoring
- Live transaction flow
- Network metrics
- Fraud alerts

### Model Performance
- Accuracy metrics
- Confusion matrix
- Feature importance

### Audit Trail
- Complete prediction history
- Model change tracking
- Data access logs

## 🐳 Docker Deployment

### Production Deployment

```bash
# Build and start all services
docker-compose -f docker-compose.yml up -d

# View logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale api=3
```

### Custom Configuration

```bash
# Override environment variables
ETHEREUM_RPC_URL=your_rpc_url docker-compose up -d

# Use custom config
docker-compose -f docker-compose.prod.yml up -d
```

## 🧪 Testing

### Run All Tests

```bash
# Run test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### API Testing

```bash
# Test API endpoints
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/model/info
```

## 🔍 Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Dashboard status
curl http://localhost:8501/_stcore/health
```

### Logs

```bash
# View application logs
tail -f logs/defi_fraud_detection.log

# Docker logs
docker-compose logs -f api
docker-compose logs -f dashboard
```

## 🚨 Troubleshooting

### Common Issues

1. **API Connection Failed**
   ```bash
   # Check if API is running
   curl http://localhost:8000/health
   
   # Check logs
   docker-compose logs api
   ```

2. **Model Not Loaded**
   ```bash
   # Train a new model
   curl -X POST "http://localhost:8000/api/v1/model/train" \
     -H "Content-Type: application/json" \
     -d '{"data_path": "data/processed/train_features.csv"}'
   ```

3. **Dashboard Not Loading**
   ```bash
   # Check Streamlit logs
   docker-compose logs dashboard
   
   # Restart dashboard
   docker-compose restart dashboard
   ```

### Performance Issues

1. **High Memory Usage**
   ```bash
   # Monitor resource usage
   docker stats
   
   # Adjust memory limits in docker-compose.yml
   ```

2. **Slow Predictions**
   ```bash
   # Check model performance
   curl http://localhost:8000/api/v1/model/info
   
   # Consider model optimization
   ```

## 📈 Scaling

### Horizontal Scaling

```bash
# Scale API instances
docker-compose up -d --scale api=5

# Add load balancer
docker-compose -f docker-compose.yml -f docker-compose.scale.yml up -d
```

### Database Scaling

```bash
# Add PostgreSQL clustering
docker-compose -f docker-compose.yml -f docker-compose.db.yml up -d
```

## 🔐 Security

### Production Security

1. **Environment Variables**
   ```bash
   # Use secure environment variables
   export SECRET_KEY=your_secure_key
   export ENCRYPTION_KEY=your_encryption_key
   ```

2. **Network Security**
   ```bash
   # Use reverse proxy
   docker-compose -f docker-compose.yml -f docker-compose.secure.yml up -d
   ```

3. **Access Control**
   ```bash
   # Enable authentication
   export ENABLE_AUTH=true
   export ADMIN_USERS=admin@example.com
   ```

## 📚 Next Steps

1. **Customize Features**: Modify feature engineering in `src/data_pipeline/feature_engineer.py`
2. **Add Models**: Implement new ML models in `src/model/fraud_detector.py`
3. **Extend API**: Add new endpoints in `src/api/endpoints.py`
4. **Enhance Dashboard**: Customize dashboard in `src/dashboard/app.py`
5. **Deploy**: Use the provided Docker configuration for production deployment

## 🆘 Support

- **Documentation**: Check `README.md` for detailed documentation
- **Issues**: Report bugs and feature requests in the repository
- **Community**: Join our community discussions

---

**Happy Fraud Detection! 🔍✨** 