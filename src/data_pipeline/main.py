"""
Main data pipeline orchestrator for DeFi fraud detection.
"""

import time
import schedule
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from .blockchain_collector import BlockchainCollector
from .feature_engineer import FeatureEngineer
from .data_processor import DataProcessor
from ..utils.logger import LoggerMixin, setup_logger
from ..utils.config import config


class DataPipeline(LoggerMixin):
    """Main data pipeline orchestrator."""
    
    def __init__(self):
        """Initialize the data pipeline."""
        super().__init__()
        
        # Initialize components
        self.collector = BlockchainCollector()
        self.feature_engineer = FeatureEngineer()
        self.data_processor = DataProcessor()
        
        # Pipeline settings
        self.collection_interval = config.get('data_pipeline.collection.collection_interval', 300)
        self.hours_back = config.get('data_pipeline.processing.feature_window_hours', 24)
        
        # Pipeline state
        self.last_run = None
        self.is_running = False
        
        self.log_info("Data pipeline initialized", 
                     collection_interval=self.collection_interval,
                     hours_back=self.hours_back)
    
    def run_pipeline(self, hours_back: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete data pipeline.
        
        Args:
            hours_back: Number of hours to look back for data collection
            
        Returns:
            Pipeline results dictionary
        """
        if self.is_running:
            self.log_warning("Pipeline already running")
            return {}
        
        self.is_running = True
        start_time = datetime.now()
        
        try:
            self.log_info("Starting data pipeline", hours_back=hours_back or self.hours_back)
            
            # Step 1: Collect blockchain data
            self.log_info("Step 1: Collecting blockchain data")
            transactions = self.collector.collect_recent_transactions(hours_back or self.hours_back)
            
            if not transactions:
                self.log_warning("No transactions collected")
                return {'status': 'no_data', 'transactions': 0}
            
            # Step 2: Save raw data
            self.log_info("Step 2: Saving raw data")
            raw_data_path = self.data_processor.save_raw_data(transactions)
            
            # Step 3: Engineer features
            self.log_info("Step 3: Engineering features")
            features_df, metadata = self.feature_engineer.engineer_features_batch(transactions)
            
            if features_df.empty:
                self.log_warning("No features created")
                return {'status': 'no_features', 'transactions': len(transactions)}
            
            # Step 4: Preprocess data
            self.log_info("Step 4: Preprocessing data")
            processed_df = self.data_processor.preprocess_data(features_df)
            
            # Step 5: Create train/test split
            self.log_info("Step 5: Creating train/test split")
            train_df, test_df = self.data_processor.create_train_test_split(processed_df)
            
            # Step 6: Save processed data
            self.log_info("Step 6: Saving processed data")
            processed_data_path = self.data_processor.save_processed_data(processed_df)
            train_data_path = self.data_processor.save_processed_data(train_df, 'train_features.csv')
            test_data_path = self.data_processor.save_processed_data(test_df, 'test_features.csv')
            
            # Step 7: Save metadata
            self.log_info("Step 7: Saving metadata")
            metadata.update({
                'pipeline_run_time': start_time.isoformat(),
                'processing_duration': (datetime.now() - start_time).total_seconds(),
                'raw_data_path': raw_data_path,
                'processed_data_path': processed_data_path,
                'train_data_path': train_data_path,
                'test_data_path': test_data_path,
                'train_shape': train_df.shape,
                'test_shape': test_df.shape
            })
            metadata_path = self.data_processor.save_metadata(metadata)
            
            # Step 8: Generate data summary
            self.log_info("Step 8: Generating data summary")
            data_summary = self.data_processor.get_data_summary(processed_df)
            
            # Pipeline results
            results = {
                'status': 'success',
                'pipeline_run_time': start_time.isoformat(),
                'processing_duration': (datetime.now() - start_time).total_seconds(),
                'transactions_collected': len(transactions),
                'features_created': len(features_df),
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'feature_count': len(processed_df.columns),
                'raw_data_path': raw_data_path,
                'processed_data_path': processed_data_path,
                'train_data_path': train_data_path,
                'test_data_path': test_data_path,
                'metadata_path': metadata_path,
                'data_summary': data_summary
            }
            
            self.last_run = start_time
            self.log_info("Data pipeline completed successfully", 
                         duration=results['processing_duration'],
                         transactions=results['transactions_collected'],
                         features=results['features_created'])
            
            return results
            
        except Exception as e:
            self.log_error("Data pipeline failed", error=str(e))
            return {
                'status': 'error',
                'error': str(e),
                'pipeline_run_time': start_time.isoformat(),
                'processing_duration': (datetime.now() - start_time).total_seconds()
            }
        
        finally:
            self.is_running = False
    
    def run_continuous_pipeline(self, callback: Optional[callable] = None):
        """
        Run the pipeline continuously at scheduled intervals.
        
        Args:
            callback: Optional callback function to call after each run
        """
        self.log_info("Starting continuous pipeline", 
                     interval_seconds=self.collection_interval)
        
        def pipeline_job():
            try:
                results = self.run_pipeline()
                if callback:
                    callback(results)
            except Exception as e:
                self.log_error("Continuous pipeline job failed", error=str(e))
        
        # Schedule the job
        schedule.every(self.collection_interval).seconds.do(pipeline_job)
        
        # Run the first job immediately
        pipeline_job()
        
        # Keep running
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                self.log_info("Continuous pipeline stopped by user")
                break
            except Exception as e:
                self.log_error("Continuous pipeline error", error=str(e))
                time.sleep(60)  # Wait before retrying
    
    def run_single_transaction_pipeline(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run pipeline for a single transaction (for real-time processing).
        
        Args:
            transaction: Single transaction data
            
        Returns:
            Pipeline results for single transaction
        """
        try:
            self.log_info("Processing single transaction", 
                         tx_hash=transaction.get('hash', ''))
            
            # Validate transaction
            is_valid, errors = self.collector.validator.validate_transaction_data(transaction)
            if not is_valid:
                self.log_warning("Invalid transaction", errors=errors)
                return {'status': 'invalid_transaction', 'errors': errors}
            
            # Engineer features for single transaction
            features = self.feature_engineer.create_fraud_features(
                transaction, {}, {}  # Empty address and network features for single transaction
            )
            
            if not features:
                return {'status': 'no_features_created'}
            
            # Convert to DataFrame
            features_df = pd.DataFrame([features])
            
            # Preprocess
            processed_df = self.data_processor.preprocess_data(features_df)
            
            results = {
                'status': 'success',
                'transaction_hash': transaction.get('hash', ''),
                'features': features,
                'processed_features': processed_df.to_dict('records')[0] if not processed_df.empty else {},
                'processing_time': datetime.now().isoformat()
            }
            
            self.log_info("Single transaction processed", 
                         tx_hash=results['transaction_hash'])
            
            return results
            
        except Exception as e:
            self.log_error("Single transaction pipeline failed", 
                          error=str(e), 
                          transaction=transaction)
            return {'status': 'error', 'error': str(e)}
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'is_running': self.is_running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'collection_interval': self.collection_interval,
            'hours_back': self.hours_back,
            'collector_connected': self.collector.check_connection(),
            'network_stats': self.collector.get_network_stats()
        }
    
    def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """
        Clean up old data files.
        
        Args:
            days_old: Files older than this many days will be deleted
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            raw_deleted = self.data_processor.cleanup_old_files(
                self.data_processor.raw_data_path, days_old
            )
            processed_deleted = self.data_processor.cleanup_old_files(
                self.data_processor.processed_data_path, days_old
            )
            
            results = {
                'raw_files_deleted': raw_deleted,
                'processed_files_deleted': processed_deleted,
                'total_files_deleted': raw_deleted + processed_deleted
            }
            
            self.log_info("Data cleanup completed", **results)
            return results
            
        except Exception as e:
            self.log_error("Data cleanup failed", error=str(e))
            return {'raw_files_deleted': 0, 'processed_files_deleted': 0, 'total_files_deleted': 0}


def main():
    """Main function to run the data pipeline."""
    # Setup logging
    logger = setup_logger(
        name="data_pipeline",
        level=config.get('logging.level', 'INFO'),
        log_file=config.get('logging.file_path', 'logs/data_pipeline.log')
    )
    
    # Initialize and run pipeline
    pipeline = DataPipeline()
    
    try:
        # Run pipeline once
        results = pipeline.run_pipeline()
        logger.info("Pipeline run completed", results=results)
        
    except Exception as e:
        logger.error("Pipeline run failed", error=str(e))
        raise


if __name__ == "__main__":
    main() 