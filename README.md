# DeFi Fraud Detection MLOps Pipeline

A comprehensive MLOps pipeline for detecting fraud in Decentralized Finance (DeFi) transactions with a private-by-design approach and modern Streamlit dashboard.

## 🏗️ Project Architecture

This project implements a continuous MLOps pipeline with three main components:

### 1. Data Pipeline & Feature Engineering ⛓️
- **Data Source**: Connects to Ethereum blockchain for transaction data
- **Feature Engineering**: Processes raw blockchain data into ML-ready features
- **Data Validation**: Ensures data quality and consistency

### 2. Model Training & API Deployment 🤖
- **Model Training**: Scikit-learn based fraud detection model
- **API Deployment**: FastAPI endpoint with Docker containerization
- **Model Versioning**: Tracks model performance and versions

### 3. Streamlit Dashboard 📊
- **Fraud Monitoring**: Real-time fraud detection metrics
- **Auditability**: Complete audit trail for compliance
- **Model Management**: Version comparison and performance tracking

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker
- Ethereum node access (or use public APIs)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd DeFi_Fraud_Detection_MLOps_Pipeline
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Run the pipeline**
```bash
# Start the data pipeline
python src/data_pipeline/main.py

# Start the API server
python src/api/main.py

# Start the dashboard
streamlit run src/dashboard/app.py
```

## 📁 Project Structure

```
DeFi_Fraud_Detection_MLOps_Pipeline/
├── src/
│   ├── data_pipeline/          # Data collection and processing
│   ├── model/                  # ML model training and inference
│   ├── api/                    # FastAPI deployment
│   ├── dashboard/              # Streamlit dashboard
│   └── utils/                  # Shared utilities
├── tests/                      # Unit and integration tests
├── docker/                     # Docker configurations
├── configs/                    # Configuration files
├── data/                       # Data storage
├── models/                     # Trained models
├── logs/                       # Application logs
└── docs/                       # Documentation
```

## 🔧 Configuration

The project uses configuration files for different environments:
- `configs/config.yaml`: Main configuration
- `configs/model_config.yaml`: Model hyperparameters
- `configs/api_config.yaml`: API settings

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📊 Monitoring

The dashboard provides real-time monitoring of:
- Fraud detection accuracy
- API response times
- Model drift detection
- Transaction volume metrics

## 🔒 Security & Privacy

- **Private-by-Design**: No sensitive data is stored
- **Audit Trail**: Complete traceability of predictions
- **Data Encryption**: All data in transit is encrypted
- **Access Control**: Role-based access to dashboard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions, please open an issue in the repository. 