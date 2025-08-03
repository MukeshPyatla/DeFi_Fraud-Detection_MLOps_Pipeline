"""
Unit tests for data pipeline components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from src.data_pipeline.blockchain_collector import BlockchainCollector
from src.data_pipeline.feature_engineer import FeatureEngineer
from src.data_pipeline.data_processor import DataProcessor
from src.data_pipeline.main import DataPipeline


class TestBlockchainCollector:
    """Test blockchain collector functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = BlockchainCollector()
    
    def test_initialization(self):
        """Test collector initialization."""
        assert self.collector is not None
        assert hasattr(self.collector, 'w3')
        assert hasattr(self.collector, 'validator')
    
    @patch('src.data_pipeline.blockchain_collector.Web3')
    def test_connection_check(self, mock_web3):
        """Test connection check functionality."""
        mock_web3_instance = Mock()
        mock_web3_instance.is_connected.return_value = True
        mock_web3.HTTPProvider.return_value = mock_web3_instance
        
        collector = BlockchainCollector()
        assert collector.check_connection() is True
    
    def test_validate_transaction_data(self):
        """Test transaction data validation."""
        valid_transaction = {
            'hash': '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
            'from_address': '0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6',
            'to_address': '0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6',
            'value': 1000000000000000000,  # 1 ETH in Wei
            'gas_price': 20000000000,  # 20 Gwei in Wei
            'gas_used': 21000,
            'block_number': 15000000,
            'timestamp': 1640995200
        }
        
        is_valid, errors = self.collector.validator.validate_transaction_data(valid_transaction)
        assert is_valid is True
        assert len(errors) == 0


class TestFeatureEngineer:
    """Test feature engineering functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.feature_engineer = FeatureEngineer()
    
    def test_initialization(self):
        """Test feature engineer initialization."""
        assert self.feature_engineer is not None
        assert hasattr(self.feature_engineer, 'validator')
    
    def test_extract_basic_features(self):
        """Test basic feature extraction."""
        transaction = {
            'hash': '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
            'from_address': '0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6',
            'to_address': '0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6',
            'value': 1000000000000000000,  # 1 ETH in Wei
            'gas_price': 20000000000,  # 20 Gwei in Wei
            'gas_used': 21000,
            'block_number': 15000000,
            'timestamp': 1640995200,
            'input': '0x'
        }
        
        features = self.feature_engineer.extract_basic_features(transaction)
        
        assert 'transaction_amount' in features
        assert 'gas_price' in features
        assert 'gas_used' in features
        assert features['transaction_amount'] == 1.0  # 1 ETH
        assert features['gas_price'] == 20.0  # 20 Gwei
    
    def test_calculate_risk_score(self):
        """Test risk score calculation."""
        features = {
            'gas_price_anomaly': 2.5,
            'transaction_value_anomaly': 6.0,
            'sender_transaction_count': 2,
            'is_contract_interaction': True,
            'transaction_frequency_anomaly': 15,
            'sender_to_receiver_ratio': 150,
            'transaction_amount': 150
        }
        
        risk_score = self.feature_engineer._calculate_risk_score(features)
        assert 0 <= risk_score <= 1
        assert risk_score > 0.5  # Should be high risk with these features


class TestDataProcessor:
    """Test data processor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()
    
    def test_initialization(self):
        """Test data processor initialization."""
        assert self.processor is not None
        assert hasattr(self.processor, 'validator')
        assert hasattr(self.processor, 'encryption')
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        # Create sample data with issues
        data = pd.DataFrame({
            'transaction_amount': [1.0, 2.0, np.nan, 4.0, np.inf],
            'gas_price': [20.0, 25.0, 30.0, np.nan, 35.0],
            'gas_used': [21000, 25000, 30000, 35000, 40000],
            'risk_score': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        processed_data = self.processor.preprocess_data(data)
        
        # Check that issues are fixed
        assert not processed_data.isnull().any().any()
        assert not np.isinf(processed_data.select_dtypes(include=[np.number])).any().any()
        assert len(processed_data) > 0
    
    def test_create_train_test_split(self):
        """Test train/test split creation."""
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'risk_score': np.random.rand(100)
        })
        
        train_df, test_df = self.processor.create_train_test_split(data, test_size=0.2)
        
        assert len(train_df) + len(test_df) == len(data)
        assert len(test_df) / len(data) == pytest.approx(0.2, rel=0.1)
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        data = pd.DataFrame({
            'numeric_feature': [1, 2, 3, 4, 5],
            'categorical_feature': ['A', 'B', 'A', 'B', 'A'],
            'risk_score': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        summary = self.processor.get_data_summary(data)
        
        assert 'shape' in summary
        assert 'columns' in summary
        assert 'numeric_summary' in summary
        assert 'categorical_summary' in summary


class TestDataPipeline:
    """Test main data pipeline functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = DataPipeline()
    
    def test_initialization(self):
        """Test pipeline initialization."""
        assert self.pipeline is not None
        assert hasattr(self.pipeline, 'collector')
        assert hasattr(self.pipeline, 'feature_engineer')
        assert hasattr(self.pipeline, 'data_processor')
    
    @patch('src.data_pipeline.blockchain_collector.BlockchainCollector')
    def test_pipeline_status(self, mock_collector):
        """Test pipeline status retrieval."""
        mock_collector_instance = Mock()
        mock_collector_instance.check_connection.return_value = True
        mock_collector_instance.get_network_stats.return_value = {
            'latest_block': 15000000,
            'gas_price': 20000000000
        }
        
        self.pipeline.collector = mock_collector_instance
        
        status = self.pipeline.get_pipeline_status()
        
        assert 'is_running' in status
        assert 'collector_connected' in status
        assert 'network_stats' in status
    
    def test_cleanup_old_data(self):
        """Test old data cleanup."""
        # This test would require actual file system operations
        # For now, just test the method exists and doesn't raise errors
        result = self.pipeline.cleanup_old_data(days_old=30)
        assert isinstance(result, dict)
        assert 'total_files_deleted' in result


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline flow."""
        # This would be a comprehensive integration test
        # For now, just verify components can be instantiated together
        pipeline = DataPipeline()
        assert pipeline is not None
        
        # Test that all components are properly initialized
        assert pipeline.collector is not None
        assert pipeline.feature_engineer is not None
        assert pipeline.data_processor is not None


# Performance tests
class TestPerformance:
    """Performance tests for the pipeline."""
    
    def test_large_dataset_processing(self):
        """Test processing of large datasets."""
        # Create large dataset
        large_data = pd.DataFrame({
            'feature1': np.random.randn(10000),
            'feature2': np.random.randn(10000),
            'feature3': np.random.randn(10000),
            'risk_score': np.random.rand(10000)
        })
        
        processor = DataProcessor()
        
        # Time the preprocessing
        import time
        start_time = time.time()
        processed_data = processor.preprocess_data(large_data)
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time (e.g., 10 seconds)
        assert processing_time < 10.0
        assert len(processed_data) == len(large_data)


if __name__ == "__main__":
    pytest.main([__file__]) 