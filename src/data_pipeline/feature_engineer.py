"""
Feature engineering for blockchain transaction data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import structlog

from ..utils.logger import LoggerMixin
from ..utils.config import config
from ..utils.validators import DataValidator


class FeatureEngineer(LoggerMixin):
    """Feature engineering for blockchain transaction data."""
    
    def __init__(self):
        """Initialize feature engineer."""
        super().__init__()
        self.validator = DataValidator()
        
        # Feature engineering settings
        self.feature_window_hours = config.get('data_pipeline.processing.feature_window_hours', 24)
        self.min_transaction_value = config.get('data_pipeline.processing.min_transaction_value', 0.01)
        self.max_transaction_value = config.get('data_pipeline.processing.max_transaction_value', 1000)
        
        # Initialize feature caches
        self.address_stats = defaultdict(dict)
        self.network_stats = {}
        
        self.log_info("Feature engineer initialized")
    
    def extract_basic_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract basic features from a single transaction.
        
        Args:
            transaction: Transaction data dictionary
            
        Returns:
            Dictionary of basic features
        """
        features = {}
        
        try:
            # Basic transaction features
            features['transaction_amount'] = float(transaction.get('value', 0)) / 1e18  # Convert from Wei to ETH
            features['gas_price'] = float(transaction.get('gas_price', 0)) / 1e9  # Convert to Gwei
            features['gas_used'] = float(transaction.get('gas_used', 0))
            features['block_time'] = float(transaction.get('timestamp', 0))
            
            # Transaction type features
            features['is_contract_interaction'] = len(transaction.get('input', '')) > 2
            features['input_data_length'] = len(transaction.get('input', ''))
            
            # Address features
            features['from_address'] = transaction.get('from_address', '')
            features['to_address'] = transaction.get('to_address', '')
            
            # Gas efficiency
            if features['gas_used'] > 0:
                features['gas_efficiency'] = features['transaction_amount'] / features['gas_used']
            else:
                features['gas_efficiency'] = 0
            
            # Value in USD (approximate - you would need price data for accurate conversion)
            features['value_usd'] = features['transaction_amount'] * 2000  # Rough ETH price estimate
            
            self.log_debug("Basic features extracted", 
                          tx_hash=transaction.get('hash', ''),
                          features_count=len(features))
            
            return features
            
        except Exception as e:
            self.log_error("Failed to extract basic features", 
                          error=str(e), 
                          transaction=transaction)
            return {}
    
    def calculate_address_features(self, transactions: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate address-based features from transaction history.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Dictionary mapping addresses to their features
        """
        address_features = defaultdict(lambda: {
            'total_sent': 0.0,
            'total_received': 0.0,
            'transaction_count': 0,
            'avg_transaction_value': 0.0,
            'max_transaction_value': 0.0,
            'min_transaction_value': float('inf'),
            'last_transaction_time': 0,
            'first_transaction_time': float('inf'),
            'unique_recipients': set(),
            'unique_senders': set()
        })
        
        try:
            for tx in transactions:
                from_addr = tx.get('from_address', '')
                to_addr = tx.get('to_address', '')
                value = float(tx.get('value', 0)) / 1e18
                timestamp = float(tx.get('timestamp', 0))
                
                # Update sender features
                if from_addr:
                    addr_features = address_features[from_addr]
                    addr_features['total_sent'] += value
                    addr_features['transaction_count'] += 1
                    addr_features['max_transaction_value'] = max(addr_features['max_transaction_value'], value)
                    addr_features['min_transaction_value'] = min(addr_features['min_transaction_value'], value)
                    addr_features['last_transaction_time'] = max(addr_features['last_transaction_time'], timestamp)
                    addr_features['first_transaction_time'] = min(addr_features['first_transaction_time'], timestamp)
                    addr_features['unique_recipients'].add(to_addr)
                
                # Update recipient features
                if to_addr:
                    addr_features = address_features[to_addr]
                    addr_features['total_received'] += value
                    addr_features['unique_senders'].add(from_addr)
            
            # Convert sets to counts and calculate averages
            for addr, features in address_features.items():
                if features['transaction_count'] > 0:
                    features['avg_transaction_value'] = features['total_sent'] / features['transaction_count']
                features['unique_recipients_count'] = len(features['unique_recipients'])
                features['unique_senders_count'] = len(features['unique_senders'])
                
                # Remove sets to make JSON serializable
                del features['unique_recipients']
                del features['unique_senders']
                
                # Handle infinite values
                if features['min_transaction_value'] == float('inf'):
                    features['min_transaction_value'] = 0.0
                if features['first_transaction_time'] == float('inf'):
                    features['first_transaction_time'] = 0.0
            
            self.log_info("Address features calculated", 
                         unique_addresses=len(address_features))
            
            return dict(address_features)
            
        except Exception as e:
            self.log_error("Failed to calculate address features", error=str(e))
            return {}
    
    def calculate_network_features(self, transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate network-level features.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Dictionary of network features
        """
        network_features = {}
        
        try:
            if not transactions:
                return network_features
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(transactions)
            df['value_eth'] = df['value'].astype(float) / 1e18
            df['gas_price_gwei'] = df['gas_price'].astype(float) / 1e9
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Network congestion features
            network_features['avg_gas_price'] = df['gas_price_gwei'].mean()
            network_features['max_gas_price'] = df['gas_price_gwei'].max()
            network_features['min_gas_price'] = df['gas_price_gwei'].min()
            network_features['gas_price_std'] = df['gas_price_gwei'].std()
            
            # Transaction volume features
            network_features['total_transaction_volume'] = df['value_eth'].sum()
            network_features['avg_transaction_value'] = df['value_eth'].mean()
            network_features['max_transaction_value'] = df['value_eth'].max()
            network_features['transaction_value_std'] = df['value_eth'].std()
            
            # Time-based features
            network_features['transaction_frequency'] = len(transactions) / self.feature_window_hours
            network_features['time_span_hours'] = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            
            # Gas usage features
            network_features['avg_gas_used'] = df['gas_used'].mean()
            network_features['total_gas_used'] = df['gas_used'].sum()
            
            # Contract interaction features
            network_features['contract_interaction_rate'] = (df['input'].str.len() > 2).mean()
            
            self.log_info("Network features calculated", 
                         features_count=len(network_features))
            
            return network_features
            
        except Exception as e:
            self.log_error("Failed to calculate network features", error=str(e))
            return {}
    
    def create_fraud_features(self, transaction: Dict[str, Any], 
                            address_features: Dict[str, Dict[str, float]],
                            network_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Create fraud detection features for a transaction.
        
        Args:
            transaction: Transaction data
            address_features: Address-based features
            network_features: Network-level features
            
        Returns:
            Dictionary of fraud detection features
        """
        fraud_features = {}
        
        try:
            # Basic transaction features
            basic_features = self.extract_basic_features(transaction)
            fraud_features.update(basic_features)
            
            # Address-based fraud indicators
            from_addr = transaction.get('from_address', '')
            to_addr = transaction.get('to_address', '')
            
            if from_addr in address_features:
                addr_features = address_features[from_addr]
                fraud_features['sender_total_sent'] = addr_features.get('total_sent', 0)
                fraud_features['sender_transaction_count'] = addr_features.get('transaction_count', 0)
                fraud_features['sender_avg_transaction_value'] = addr_features.get('avg_transaction_value', 0)
                fraud_features['sender_unique_recipients'] = addr_features.get('unique_recipients_count', 0)
                fraud_features['sender_account_age_hours'] = (
                    addr_features.get('last_transaction_time', 0) - 
                    addr_features.get('first_transaction_time', 0)
                ) / 3600
            else:
                fraud_features['sender_total_sent'] = 0
                fraud_features['sender_transaction_count'] = 0
                fraud_features['sender_avg_transaction_value'] = 0
                fraud_features['sender_unique_recipients'] = 0
                fraud_features['sender_account_age_hours'] = 0
            
            if to_addr in address_features:
                addr_features = address_features[to_addr]
                fraud_features['receiver_total_received'] = addr_features.get('total_received', 0)
                fraud_features['receiver_unique_senders'] = addr_features.get('unique_senders_count', 0)
            else:
                fraud_features['receiver_total_received'] = 0
                fraud_features['receiver_unique_senders'] = 0
            
            # Network-based fraud indicators
            fraud_features['network_avg_gas_price'] = network_features.get('avg_gas_price', 0)
            fraud_features['network_transaction_frequency'] = network_features.get('transaction_frequency', 0)
            fraud_features['network_contract_interaction_rate'] = network_features.get('contract_interaction_rate', 0)
            
            # Anomaly indicators
            fraud_features['gas_price_anomaly'] = (
                fraud_features['gas_price'] / network_features.get('avg_gas_price', 1)
                if network_features.get('avg_gas_price', 0) > 0 else 1
            )
            
            fraud_features['transaction_value_anomaly'] = (
                fraud_features['transaction_amount'] / network_features.get('avg_transaction_value', 1)
                if network_features.get('avg_transaction_value', 0) > 0 else 1
            )
            
            # Behavioral features
            fraud_features['sender_to_receiver_ratio'] = (
                fraud_features['sender_total_sent'] / fraud_features['receiver_total_received']
                if fraud_features['receiver_total_received'] > 0 else 0
            )
            
            fraud_features['transaction_frequency_anomaly'] = (
                fraud_features['sender_transaction_count'] / self.feature_window_hours
                - network_features.get('transaction_frequency', 0)
            )
            
            # Risk scores
            fraud_features['risk_score'] = self._calculate_risk_score(fraud_features)
            
            # Validate features
            is_valid, errors = self.validator.validate_feature_data(fraud_features)
            if not is_valid:
                self.log_warning("Invalid fraud features", errors=errors)
            
            self.log_debug("Fraud features created", 
                          tx_hash=transaction.get('hash', ''),
                          features_count=len(fraud_features))
            
            return fraud_features
            
        except Exception as e:
            self.log_error("Failed to create fraud features", 
                          error=str(e), 
                          transaction=transaction)
            return {}
    
    def _calculate_risk_score(self, features: Dict[str, Any]) -> float:
        """
        Calculate a risk score based on features.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Risk score between 0 and 1
        """
        risk_score = 0.0
        
        try:
            # High gas price anomaly
            if features.get('gas_price_anomaly', 1) > 2:
                risk_score += 0.2
            
            # High transaction value anomaly
            if features.get('transaction_value_anomaly', 1) > 5:
                risk_score += 0.2
            
            # New sender (low transaction count)
            if features.get('sender_transaction_count', 0) < 3:
                risk_score += 0.15
            
            # Contract interaction
            if features.get('is_contract_interaction', False):
                risk_score += 0.1
            
            # High frequency transactions
            if features.get('transaction_frequency_anomaly', 0) > 10:
                risk_score += 0.15
            
            # Unusual sender-to-receiver ratio
            if features.get('sender_to_receiver_ratio', 0) > 100:
                risk_score += 0.1
            
            # Very high transaction value
            if features.get('transaction_amount', 0) > 100:
                risk_score += 0.1
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            self.log_error("Failed to calculate risk score", error=str(e))
            return 0.0
    
    def engineer_features_batch(self, transactions: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Engineer features for a batch of transactions.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Tuple of (feature DataFrame, metadata)
        """
        try:
            self.log_info("Starting batch feature engineering", 
                         transaction_count=len(transactions))
            
            # Calculate address and network features
            address_features = self.calculate_address_features(transactions)
            network_features = self.calculate_network_features(transactions)
            
            # Create fraud features for each transaction
            fraud_features_list = []
            for tx in transactions:
                fraud_features = self.create_fraud_features(tx, address_features, network_features)
                if fraud_features:
                    fraud_features['transaction_hash'] = tx.get('hash', '')
                    fraud_features_list.append(fraud_features)
            
            # Convert to DataFrame
            if fraud_features_list:
                df = pd.DataFrame(fraud_features_list)
                
                # Validate DataFrame
                is_valid, errors = self.validator.validate_dataframe(df)
                if not is_valid:
                    self.log_warning("Feature DataFrame validation failed", errors=errors)
                
                # Create metadata
                metadata = {
                    'total_transactions': len(transactions),
                    'processed_transactions': len(fraud_features_list),
                    'feature_count': len(df.columns),
                    'address_count': len(address_features),
                    'network_features': network_features,
                    'processing_timestamp': datetime.now().isoformat()
                }
                
                self.log_info("Batch feature engineering completed", 
                             processed_count=len(fraud_features_list),
                             feature_count=len(df.columns))
                
                return df, metadata
            else:
                self.log_warning("No valid features created from transactions")
                return pd.DataFrame(), {}
                
        except Exception as e:
            self.log_error("Failed to engineer features batch", error=str(e))
            return pd.DataFrame(), {}
    
    def save_features(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save engineered features to file.
        
        Args:
            df: Feature DataFrame
            filename: Output filename
        """
        try:
            df.to_csv(filename, index=False)
            self.log_info("Features saved", 
                         filename=filename, 
                         shape=df.shape)
        except Exception as e:
            self.log_error("Failed to save features", 
                          error=str(e), 
                          filename=filename)
            raise 