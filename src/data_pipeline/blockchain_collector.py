"""
Blockchain data collector for Ethereum transaction data.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Generator
from datetime import datetime, timedelta
import pandas as pd
from web3 import Web3
from web3.middleware import geth_poa_middleware
import structlog

from ..utils.logger import LoggerMixin
from ..utils.config import config
from ..utils.validators import DataValidator


class BlockchainCollector(LoggerMixin):
    """Collects blockchain transaction data from Ethereum network."""
    
    def __init__(self, rpc_url: Optional[str] = None):
        """
        Initialize blockchain collector.
        
        Args:
            rpc_url: Ethereum RPC URL. If not provided, uses config.
        """
        super().__init__()
        self.rpc_url = rpc_url or config.get('data_pipeline.blockchain.rpc_url')
        self.ws_url = config.get('data_pipeline.blockchain.ws_url')
        self.chain_id = config.get('data_pipeline.blockchain.chain_id', 1)
        
        # Initialize Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        if self.chain_id != 1:  # Not mainnet
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Initialize validator
        self.validator = DataValidator()
        
        # Collection settings
        self.batch_size = config.get('data_pipeline.collection.batch_size', 1000)
        self.max_retries = config.get('data_pipeline.collection.max_retries', 3)
        self.retry_delay = config.get('data_pipeline.collection.retry_delay', 5)
        
        self.log_info("Blockchain collector initialized", 
                     rpc_url=self.rpc_url, 
                     chain_id=self.chain_id)
    
    def check_connection(self) -> bool:
        """Check if Web3 connection is working."""
        try:
            return self.w3.is_connected()
        except Exception as e:
            self.log_error("Web3 connection check failed", error=str(e))
            return False
    
    def get_latest_block_number(self) -> int:
        """Get the latest block number."""
        try:
            return self.w3.eth.block_number
        except Exception as e:
            self.log_error("Failed to get latest block number", error=str(e))
            raise
    
    def get_block_data(self, block_number: int) -> Dict[str, Any]:
        """
        Get block data for a specific block number.
        
        Args:
            block_number: Block number to retrieve
            
        Returns:
            Block data dictionary
        """
        try:
            block = self.w3.eth.get_block(block_number, full_transactions=True)
            
            # Convert block data to dictionary
            block_data = {
                'number': block.number,
                'hash': block.hash.hex(),
                'timestamp': block.timestamp,
                'gas_limit': block.gasLimit,
                'gas_used': block.gasUsed,
                'transactions': []
            }
            
            # Process transactions
            for tx in block.transactions:
                tx_data = {
                    'hash': tx.hash.hex(),
                    'from_address': tx['from'],
                    'to_address': tx['to'],
                    'value': tx['value'],
                    'gas_price': tx['gasPrice'],
                    'gas_used': tx.get('gas', 0),
                    'block_number': block_number,
                    'timestamp': block.timestamp,
                    'nonce': tx['nonce'],
                    'input': tx['input']
                }
                
                # Validate transaction data
                is_valid, errors = self.validator.validate_transaction_data(tx_data)
                if is_valid:
                    block_data['transactions'].append(tx_data)
                else:
                    self.log_warning("Invalid transaction data", 
                                   errors=errors, 
                                   tx_hash=tx_data['hash'])
            
            # Validate block data
            is_valid, errors = self.validator.validate_block_data(block_data)
            if not is_valid:
                self.log_warning("Invalid block data", errors=errors, block_number=block_number)
            
            self.log_debug("Block data retrieved", 
                          block_number=block_number, 
                          tx_count=len(block_data['transactions']))
            
            return block_data
            
        except Exception as e:
            self.log_error("Failed to get block data", 
                          error=str(e), 
                          block_number=block_number)
            raise
    
    def get_transaction_receipt(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get transaction receipt for additional data.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction receipt data
        """
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            return {
                'hash': receipt['transactionHash'].hex(),
                'gas_used': receipt['gasUsed'],
                'status': receipt['status'],
                'logs': len(receipt['logs'])
            }
        except Exception as e:
            self.log_warning("Failed to get transaction receipt", 
                           error=str(e), 
                           tx_hash=tx_hash)
            return None
    
    def collect_block_range(self, start_block: int, end_block: int) -> List[Dict[str, Any]]:
        """
        Collect data for a range of blocks.
        
        Args:
            start_block: Starting block number
            end_block: Ending block number
            
        Returns:
            List of block data dictionaries
        """
        blocks_data = []
        
        for block_number in range(start_block, end_block + 1):
            try:
                block_data = self.get_block_data(block_number)
                blocks_data.append(block_data)
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                self.log_error("Failed to collect block", 
                             error=str(e), 
                             block_number=block_number)
                continue
        
        self.log_info("Block range collection completed", 
                     start_block=start_block, 
                     end_block=end_block, 
                     blocks_collected=len(blocks_data))
        
        return blocks_data
    
    def collect_recent_transactions(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Collect recent transactions from the last N hours.
        
        Args:
            hours_back: Number of hours to look back
            
        Returns:
            List of transaction data dictionaries
        """
        try:
            latest_block = self.get_latest_block_number()
            current_time = datetime.now()
            
            # Estimate blocks to go back (assuming ~12 second block time)
            blocks_per_hour = 300  # 3600 seconds / 12 seconds per block
            blocks_to_go_back = hours_back * blocks_per_hour
            
            start_block = max(0, latest_block - blocks_to_go_back)
            
            self.log_info("Collecting recent transactions", 
                         hours_back=hours_back, 
                         start_block=start_block, 
                         end_block=latest_block)
            
            blocks_data = self.collect_block_range(start_block, latest_block)
            
            # Extract transactions from blocks
            all_transactions = []
            for block_data in blocks_data:
                all_transactions.extend(block_data['transactions'])
            
            self.log_info("Recent transactions collected", 
                         total_transactions=len(all_transactions))
            
            return all_transactions
            
        except Exception as e:
            self.log_error("Failed to collect recent transactions", error=str(e))
            raise
    
    def get_account_balance(self, address: str) -> int:
        """
        Get account balance for an address.
        
        Args:
            address: Ethereum address
            
        Returns:
            Account balance in Wei
        """
        try:
            balance = self.w3.eth.get_balance(address)
            return balance
        except Exception as e:
            self.log_error("Failed to get account balance", 
                          error=str(e), 
                          address=address)
            return 0
    
    def get_gas_price(self) -> int:
        """Get current gas price."""
        try:
            return self.w3.eth.gas_price
        except Exception as e:
            self.log_error("Failed to get gas price", error=str(e))
            return 0
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get current network statistics."""
        try:
            latest_block = self.get_latest_block_number()
            gas_price = self.get_gas_price()
            
            return {
                'latest_block': latest_block,
                'gas_price': gas_price,
                'timestamp': datetime.now().isoformat(),
                'network': 'ethereum',
                'chain_id': self.chain_id
            }
        except Exception as e:
            self.log_error("Failed to get network stats", error=str(e))
            return {}
    
    def monitor_transactions(self, callback=None) -> Generator[Dict[str, Any], None, None]:
        """
        Monitor new transactions in real-time.
        
        Args:
            callback: Optional callback function for new transactions
            
        Yields:
            New transaction data
        """
        try:
            latest_block = self.get_latest_block_number()
            
            while True:
                current_block = self.get_latest_block_number()
                
                if current_block > latest_block:
                    # New blocks available
                    for block_number in range(latest_block + 1, current_block + 1):
                        block_data = self.get_block_data(block_number)
                        
                        for tx in block_data['transactions']:
                            if callback:
                                callback(tx)
                            yield tx
                    
                    latest_block = current_block
                
                # Wait before checking again
                time.sleep(12)  # Wait for next block
                
        except KeyboardInterrupt:
            self.log_info("Transaction monitoring stopped")
        except Exception as e:
            self.log_error("Transaction monitoring failed", error=str(e))
            raise
    
    def save_transactions_to_csv(self, transactions: List[Dict[str, Any]], 
                                filename: str) -> None:
        """
        Save transactions to CSV file.
        
        Args:
            transactions: List of transaction dictionaries
            filename: Output CSV filename
        """
        try:
            df = pd.DataFrame(transactions)
            df.to_csv(filename, index=False)
            self.log_info("Transactions saved to CSV", 
                         filename=filename, 
                         transaction_count=len(transactions))
        except Exception as e:
            self.log_error("Failed to save transactions to CSV", 
                          error=str(e), 
                          filename=filename)
            raise 