"""
Configuration management for the DeFi Fraud Detection Pipeline.
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Configuration manager for the DeFi Fraud Detection Pipeline."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize configuration from YAML file and environment variables."""
        self.config_path = config_path
        self._config = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file and environment variables."""
        # Load environment variables
        load_dotenv()
        
        # Load YAML configuration
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as file:
                self._config = yaml.safe_load(file)
        
        # Override with environment variables
        self._override_from_env()
    
    def _override_from_env(self) -> None:
        """Override configuration with environment variables."""
        env_mappings = {
            'APP_NAME': ('app', 'name'),
            'APP_VERSION': ('app', 'version'),
            'ENVIRONMENT': ('app', 'environment'),
            'DEBUG': ('app', 'debug'),
            'ETHEREUM_RPC_URL': ('data_pipeline', 'blockchain', 'rpc_url'),
            'ETHEREUM_WS_URL': ('data_pipeline', 'blockchain', 'ws_url'),
            'ETHEREUM_CHAIN_ID': ('data_pipeline', 'blockchain', 'chain_id'),
            'API_HOST': ('api', 'host'),
            'API_PORT': ('api', 'port'),
            'API_WORKERS': ('api', 'workers'),
            'API_TIMEOUT': ('api', 'timeout'),
            'DASHBOARD_HOST': ('dashboard', 'host'),
            'DASHBOARD_PORT': ('dashboard', 'port'),
            'DATABASE_URL': ('database', 'url'),
            'SECRET_KEY': ('security', 'secret_key'),
            'ENCRYPTION_KEY': ('security', 'encryption_key'),
            'ENABLE_METRICS': ('api', 'monitoring', 'enable_metrics'),
            'ENABLE_LOGGING': ('api', 'monitoring', 'enable_logging'),
            'LOG_LEVEL': ('logging', 'level'),
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(config_path, env_value)
    
    def _set_nested_value(self, path: tuple, value: Any) -> None:
        """Set a nested configuration value."""
        current = self._config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert value to appropriate type
        if isinstance(value, str):
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
        
        current[path[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Get nested configuration value."""
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split('.')
        current = self._config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to YAML file."""
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)
    
    def validate(self) -> bool:
        """Validate required configuration values."""
        required_keys = [
            'app.name',
            'app.version',
            'data_pipeline.blockchain.rpc_url',
            'api.host',
            'api.port',
            'dashboard.host',
            'dashboard.port'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                raise ValueError(f"Required configuration key missing: {key}")
        
        return True


# Global configuration instance
config = Config() 