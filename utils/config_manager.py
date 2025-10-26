"""
Configuration Manager for Advanced Log Reader
Handles application settings, patterns, and configuration management.
"""

import json
import os
from pathlib import Path
import logging

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration settings for the log analysis application"""
    
    DEFAULT_CONFIG = {
        "log_level": "INFO",
        "log_file": "app.log",
        "utterance_patterns": [],
        "analysis": {
            "max_file_size_mb": 500,
            "relevance_threshold": 0.5,
            "session_timeout_seconds": 120,
            "max_sessions_display": 100
        },
        "utterance_patterns_extended": {
            "alexa": {
                "wake_words": [
                    r'\b(alexa|amazon|echo)\b',
                    r'\bwake\s+word\b',
                    r'\bactivation\s+detected\b'
                ],
                "dialog_states": [
                    "LISTENING", "THINKING", "SPEAKING", "IDLE"
                ]
            },
            "custom": []
        },
        "ai_model": {
            "retrain_interval_days": 7,
            "confidence_threshold": 0.7,
            "max_model_age_days": 30
        },
        "ui": {
            "theme": "light",
            "max_log_lines_display": 1000,
            "chart_colors": {
                "primary": "#1f77b4",
                "secondary": "#ff7f0e",
                "error": "#d62728",
                "warning": "#ff7f0e",
                "info": "#2ca02c"
            }
        },
        "export": {
            "default_format": "csv",
            "include_metadata": True,
            "timestamp_format": "%Y-%m-%d %H:%M:%S"
        }
    }

    def __init__(self):
        self.config = self.DEFAULT_CONFIG.copy()

    def load_config(self, file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The configuration file {file_path} does not exist.")

        with open(file_path, 'r') as file:
            if file_path.endswith('.json'):
                self.config.update(json.load(file))
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                if HAS_YAML and 'yaml' in globals():
                    self.config.update(yaml.safe_load(file))
                else:
                    raise ValueError("YAML support not available. Please install PyYAML or use JSON format.")
            else:
                raise ValueError("Unsupported file format. Please use JSON or YAML.")

    def merge_config(self, new_config):
        """Recursively merge configuration dictionaries"""
        self._merge_dict(self.config, new_config)
    
    def _merge_dict(self, default, override):
        """Recursively merge dictionaries"""
        for key, value in override.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_dict(default[key], value)
            else:
                default[key] = value

    def get_config(self, key, default=None):
        return self.config.get(key, default)
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation (e.g., 'analysis.max_file_size_mb')"""
        try:
            keys = key_path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set_config(self, key, value):
        self.config[key] = value
    
    def set(self, key_path, value):
        """Set configuration value using dot notation"""
        try:
            keys = key_path.split('.')
            config = self.config
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            config[keys[-1]] = value
        except Exception as e:
            logger.error(f"Error setting config value {key_path}: {e}")
    
    def save_config(self, file_path):
        """Save current configuration to file"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
    
    def get_analysis_config(self):
        """Get analysis-specific configuration"""
        return self.config.get('analysis', {})
    
    def get_utterance_patterns(self):
        """Get utterance pattern configuration"""
        return self.config.get('utterance_patterns_extended', {})
    
    def get_ai_config(self):
        """Get AI model configuration"""
        return self.config.get('ai_model', {})
    
    def get_ui_config(self):
        """Get UI configuration"""
        return self.config.get('ui', {})
    
    def add_custom_pattern(self, pattern):
        """Add a custom utterance pattern"""
        if 'utterance_patterns_extended' not in self.config:
            self.config['utterance_patterns_extended'] = {}
        if 'custom' not in self.config['utterance_patterns_extended']:
            self.config['utterance_patterns_extended']['custom'] = []
        
        if pattern not in self.config['utterance_patterns_extended']['custom']:
            self.config['utterance_patterns_extended']['custom'].append(pattern)
            logger.info(f"Added custom pattern: {pattern}")

# Global config instance
_config_manager = None

def get_config_manager():
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config(key_path, default=None):
    """Convenience function to get configuration values"""
    return get_config_manager().get(key_path, default)

def set_config(key_path, value):
    """Convenience function to set configuration values"""
    return get_config_manager().set(key_path, value)
