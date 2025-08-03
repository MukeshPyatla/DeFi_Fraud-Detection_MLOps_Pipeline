#!/usr/bin/env python3
"""
Setup script for DeFi Fraud Detection MLOps Pipeline.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "logs",
        "tests",
        "docs"
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}")

def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    # Check if pip is available
    if not shutil.which("pip"):
        print("❌ pip not found. Please install Python and pip first.")
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def setup_environment():
    """Set up environment variables."""
    print("🔧 Setting up environment...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file from template...")
        shutil.copy("env.example", ".env")
        print("✅ Created .env file")
        print("⚠️  Please edit .env file with your configuration")
    else:
        print("✅ .env file already exists")

def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but continuing with setup")
        return False
    
    return True

def build_docker():
    """Build Docker images."""
    print("🐳 Building Docker images...")
    
    if not shutil.which("docker"):
        print("⚠️  Docker not found. Skipping Docker build.")
        return True
    
    if not run_command("docker build -t defi-fraud-detection .", "Building Docker image"):
        print("⚠️  Docker build failed, but continuing with setup")
        return False
    
    return True

def check_system_requirements():
    """Check system requirements."""
    print("🔍 Checking system requirements...")
    
    requirements = {
        "Python": "python3",
        "pip": "pip",
        "Git": "git"
    }
    
    missing = []
    for name, command in requirements.items():
        if shutil.which(command):
            print(f"✅ {name} found")
        else:
            print(f"❌ {name} not found")
            missing.append(name)
    
    if missing:
        print(f"⚠️  Missing requirements: {', '.join(missing)}")
        print("Please install missing requirements before continuing")
        return False
    
    return True

def create_sample_data():
    """Create sample data for testing."""
    print("📊 Creating sample data...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample transaction data
        sample_transactions = []
        for i in range(100):
            transaction = {
                'hash': f'0x{i:064x}',
                'from_address': f'0x{i:040x}',
                'to_address': f'0x{(i+1):040x}',
                'value': np.random.uniform(0.01, 10.0),
                'gas_price': np.random.uniform(10, 100),
                'gas_used': np.random.randint(21000, 100000),
                'block_number': 15000000 + i,
                'timestamp': 1640995200 + i * 12
            }
            sample_transactions.append(transaction)
        
        # Save sample data
        import json
        with open('data/raw/sample_transactions.json', 'w') as f:
            json.dump(sample_transactions, f, indent=2)
        
        print("✅ Created sample transaction data")
        
    except ImportError:
        print("⚠️  Could not create sample data (pandas not available)")

def main():
    """Main setup function."""
    print("🚀 Setting up DeFi Fraud Detection MLOps Pipeline")
    print("=" * 60)
    
    # Check system requirements
    if not check_system_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Set up environment
    setup_environment()
    
    # Create sample data
    create_sample_data()
    
    # Run tests
    run_tests()
    
    # Build Docker (optional)
    build_docker()
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your configuration")
    print("2. Configure your Ethereum RPC URL")
    print("3. Run the data pipeline: python src/data_pipeline/main.py")
    print("4. Start the API: python src/api/main.py")
    print("5. Start the dashboard: streamlit run src/dashboard/app.py")
    print("\n📚 Documentation:")
    print("- README.md for detailed instructions")
    print("- API docs available at http://localhost:8000/docs")
    print("- Dashboard available at http://localhost:8501")
    print("\n🐳 Docker deployment:")
    print("- docker-compose up -d")
    print("\n🧪 Testing:")
    print("- python -m pytest tests/")

if __name__ == "__main__":
    main() 