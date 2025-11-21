"""
Configuration Loader
Load and manage configuration settings
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration

        Args:
            config_path: Path to YAML config file (optional)
        """
        # Default configuration
        self.config = self._get_defaults()

        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)

        # Override with environment variables
        self._load_from_env()

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'llm': {
                'provider': 'openai',
                'model': None,
                'temperature': 0.5,
                'max_tokens': 1000,
                'ollama_base_url': 'http://localhost:11434'
            },
            'trading': {
                'initial_capital': 100000.0,
                'max_position_size': 10.0,
                'max_exposure_pct': 0.5,
                'max_drawdown_pct': 0.15,
                'stop_loss_pct': 0.05,
                'commission': 0.001
            },
            'risk': {
                'conservative_mode': False,
                'require_high_confidence': False,
                'min_confidence_threshold': 0.6
            },
            'data': {
                'lookback_window': 20,
                'indicator_warmup': 50
            },
            'agents': {
                'market_analyst': {'temperature': 0.3, 'max_tokens': 800},
                'news_analyst': {'temperature': 0.4, 'max_tokens': 600},
                'strategy': {'temperature': 0.5, 'max_tokens': 1000},
                'cio': {'temperature': 0.4, 'max_tokens': 800},
                'risk_manager': {'temperature': 0.2, 'max_tokens': 700},
                'execution': {'temperature': 0.1, 'max_tokens': 500, 'use_llm': False}
            },
            'logging': {
                'log_dir': 'logs',
                'verbose': True,
                'save_on_completion': True
            },
            'backtest': {
                'start_date': None,
                'end_date': None,
                'start_idx': 50
            }
        }

    def load_from_file(self, config_path: str):
        """
        Load configuration from YAML file

        Args:
            config_path: Path to YAML file
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, 'r') as f:
            file_config = yaml.safe_load(f)

        # Merge with defaults (file config overrides defaults)
        self._merge_configs(self.config, file_config)

    def _merge_configs(self, base: Dict, override: Dict):
        """
        Recursively merge override config into base config

        Args:
            base: Base configuration
            override: Override configuration
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def _load_from_env(self):
        """Load configuration from environment variables"""
        # LLM API keys
        if os.getenv('OPENAI_API_KEY'):
            self.config.setdefault('api_keys', {})['openai'] = os.getenv('OPENAI_API_KEY')
        if os.getenv('ANTHROPIC_API_KEY'):
            self.config.setdefault('api_keys', {})['anthropic'] = os.getenv('ANTHROPIC_API_KEY')

        # Override provider from env
        if os.getenv('LLM_PROVIDER'):
            self.config['llm']['provider'] = os.getenv('LLM_PROVIDER')

        # Override model from env
        if os.getenv('LLM_MODEL'):
            self.config['llm']['model'] = os.getenv('LLM_MODEL')

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key_path: Configuration key path (e.g., 'llm.provider')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation

        Args:
            key_path: Configuration key path (e.g., 'llm.provider')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return self.config.copy()

    def __repr__(self) -> str:
        return f"Config(provider={self.get('llm.provider')}, capital={self.get('trading.initial_capital')})"


# Global config instance
_global_config: Optional[Config] = None


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load global configuration

    Args:
        config_path: Path to config file (optional)

    Returns:
        Config instance
    """
    global _global_config

    if _global_config is None:
        _global_config = Config(config_path)

    return _global_config


def get_config() -> Config:
    """
    Get global configuration instance

    Returns:
        Config instance
    """
    global _global_config

    if _global_config is None:
        _global_config = Config()

    return _global_config
