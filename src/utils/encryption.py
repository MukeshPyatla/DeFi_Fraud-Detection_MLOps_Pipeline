"""
Encryption utilities for secure data handling in the DeFi Fraud Detection Pipeline.
"""

import os
import base64
import hashlib
from typing import Dict, Any, Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from .logger import LoggerMixin


class DataEncryption(LoggerMixin):
    """Encryption utilities for secure data handling."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize encryption utilities.
        
        Args:
            encryption_key: Optional encryption key. If not provided, will use environment variable.
        """
        super().__init__()
        self.encryption_key = encryption_key or os.getenv('ENCRYPTION_KEY')
        if not self.encryption_key:
            self.encryption_key = self._generate_key()
        
        self.fernet = Fernet(self.encryption_key.encode())
    
    def _generate_key(self) -> str:
        """Generate a new encryption key."""
        key = Fernet.generate_key()
        return key.decode()
    
    def _derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def encrypt_data(self, data: Union[str, bytes, Dict[str, Any]]) -> str:
        """
        Encrypt data using Fernet symmetric encryption.
        
        Args:
            data: Data to encrypt (string, bytes, or dictionary)
            
        Returns:
            Base64 encoded encrypted data
        """
        try:
            if isinstance(data, dict):
                import json
                data_str = json.dumps(data)
            elif isinstance(data, bytes):
                data_str = data.decode('utf-8')
            else:
                data_str = str(data)
            
            encrypted_data = self.fernet.encrypt(data_str.encode())
            return base64.b64encode(encrypted_data).decode()
        
        except Exception as e:
            self.log_error("Encryption failed", error=str(e), data_type=type(data))
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt data using Fernet symmetric encryption.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted data as string
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode('utf-8')
        
        except Exception as e:
            self.log_error("Decryption failed", error=str(e))
            raise
    
    def encrypt_sensitive_fields(self, data: Dict[str, Any], 
                               sensitive_fields: list = None) -> Dict[str, Any]:
        """
        Encrypt sensitive fields in a dictionary while leaving others unchanged.
        
        Args:
            data: Dictionary containing data
            sensitive_fields: List of field names to encrypt
            
        Returns:
            Dictionary with sensitive fields encrypted
        """
        if sensitive_fields is None:
            sensitive_fields = ['address', 'hash', 'private_key', 'password']
        
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in data and data[field] is not None:
                try:
                    encrypted_data[field] = self.encrypt_data(data[field])
                    self.log_debug(f"Encrypted sensitive field: {field}")
                except Exception as e:
                    self.log_warning(f"Failed to encrypt field {field}", error=str(e))
        
        return encrypted_data
    
    def decrypt_sensitive_fields(self, data: Dict[str, Any], 
                               sensitive_fields: list = None) -> Dict[str, Any]:
        """
        Decrypt sensitive fields in a dictionary.
        
        Args:
            data: Dictionary containing encrypted data
            sensitive_fields: List of field names to decrypt
            
        Returns:
            Dictionary with sensitive fields decrypted
        """
        if sensitive_fields is None:
            sensitive_fields = ['address', 'hash', 'private_key', 'password']
        
        decrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in data and data[field] is not None:
                try:
                    decrypted_data[field] = self.decrypt_data(data[field])
                    self.log_debug(f"Decrypted sensitive field: {field}")
                except Exception as e:
                    self.log_warning(f"Failed to decrypt field {field}", error=str(e))
        
        return decrypted_data
    
    def hash_data(self, data: Union[str, bytes]) -> str:
        """
        Create a SHA-256 hash of data.
        
        Args:
            data: Data to hash
            
        Returns:
            SHA-256 hash as hexadecimal string
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.sha256(data).hexdigest()
    
    def anonymize_address(self, address: str) -> str:
        """
        Anonymize blockchain address by hashing it.
        
        Args:
            address: Blockchain address
            
        Returns:
            Anonymized address hash
        """
        return self.hash_data(address)
    
    def create_audit_hash(self, data: Dict[str, Any]) -> str:
        """
        Create a hash for audit trail purposes.
        
        Args:
            data: Data to hash for audit
            
        Returns:
            Audit hash
        """
        # Sort keys to ensure consistent hashing
        sorted_data = dict(sorted(data.items()))
        data_str = str(sorted_data)
        return self.hash_data(data_str)
    
    def verify_data_integrity(self, original_hash: str, data: Dict[str, Any]) -> bool:
        """
        Verify data integrity by comparing hashes.
        
        Args:
            original_hash: Original hash of the data
            data: Current data to verify
            
        Returns:
            True if data integrity is maintained
        """
        current_hash = self.create_audit_hash(data)
        return original_hash == current_hash
    
    def secure_transaction_data(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Securely process transaction data for storage/transmission.
        
        Args:
            transaction: Transaction data dictionary
            
        Returns:
            Secured transaction data
        """
        # Create a copy for processing
        secured_transaction = transaction.copy()
        
        # Anonymize addresses
        if 'from_address' in secured_transaction:
            secured_transaction['from_address'] = self.anonymize_address(
                secured_transaction['from_address']
            )
        
        if 'to_address' in secured_transaction:
            secured_transaction['to_address'] = self.anonymize_address(
                secured_transaction['to_address']
            )
        
        # Create audit hash
        secured_transaction['audit_hash'] = self.create_audit_hash(transaction)
        
        # Add encryption timestamp
        secured_transaction['encrypted_at'] = str(hashlib.sha256(
            str(transaction.get('timestamp', '')).encode()
        ).hexdigest()[:16]
        
        self.log_info("Transaction data secured", 
                     original_keys=list(transaction.keys()),
                     secured_keys=list(secured_transaction.keys()))
        
        return secured_transaction
    
    def generate_secure_id(self, data: Dict[str, Any]) -> str:
        """
        Generate a secure, unique identifier for data.
        
        Args:
            data: Data to generate ID for
            
        Returns:
            Secure unique identifier
        """
        # Create a deterministic string representation
        sorted_items = sorted(data.items())
        data_str = "|".join(f"{k}:{v}" for k, v in sorted_items)
        
        # Generate hash
        return self.hash_data(data_str)[:16]  # Use first 16 characters
    
    def mask_sensitive_data(self, data: Dict[str, Any], 
                           mask_fields: list = None) -> Dict[str, Any]:
        """
        Mask sensitive data for logging/debugging purposes.
        
        Args:
            data: Data to mask
            mask_fields: Fields to mask
            
        Returns:
            Data with sensitive fields masked
        """
        if mask_fields is None:
            mask_fields = ['address', 'hash', 'private_key', 'password', 'secret']
        
        masked_data = data.copy()
        
        for field in mask_fields:
            if field in masked_data and masked_data[field] is not None:
                value = str(masked_data[field])
                if len(value) > 8:
                    masked_data[field] = value[:4] + "*" * (len(value) - 8) + value[-4:]
                else:
                    masked_data[field] = "*" * len(value)
        
        return masked_data 