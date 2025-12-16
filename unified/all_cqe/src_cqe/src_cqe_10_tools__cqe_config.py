"""
CQE Configuration System

Centralized configuration management for the CQE system with support
for environment variables, config files, and runtime overrides.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class CQEConfig:
    """Complete CQE system configuration"""

    # System settings
    log_level: str = "INFO"
    max_atoms: int = 1000000
    validation_timeout: float = 30.0
    enable_caching: bool = True

    # E8 Lattice settings
    e8_precision: float = 1e-10
    lattice_dimension: int = 8
    max_shell_radius: float = 10.0

    # Slice settings  
    enable_all_slices: bool = True
    disabled_slices: list = field(default_factory=list)
    slice_timeout: float = 5.0

    # Validation settings
    parity_lanes: int = 64
    governance_bands: Dict[str, int] = field(default_factory=lambda: {"band8": 8, "band24": 24, "tile4096": 4096})
    energy_constraint_strict: bool = True

    # Performance settings
    max_concurrent_validations: int = 100
    memory_limit_mb: int = 8192
    cache_size_mb: int = 1024

    # Storage settings
    ledger_path: str = "./cqe_ledger.db"
    atom_storage_path: str = "./cqe_atoms/"
    backup_enabled: bool = True
    backup_interval_hours: int = 24

    # Network settings
    api_host: str = "localhost"
    api_port: int = 8888
    enable_distributed: bool = False
    cluster_nodes: list = field(default_factory=list)

    # Development settings
    debug_mode: bool = False
    profile_performance: bool = False
    enable_metrics: bool = True

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'CQEConfig':
        """Load configuration from file or environment"""

        config = cls()

        # Load from config file if provided
        if config_path and Path(config_path).exists():
            config = cls._load_from_file(config_path)
        else:
            # Try to load from default locations
            default_paths = [
                "./cqe_config.yaml",
                "./cqe_config.json", 
                "~/.cqe/config.yaml",
                "/etc/cqe/config.yaml"
            ]

            for path in default_paths:
                expanded_path = Path(path).expanduser()
                if expanded_path.exists():
                    config = cls._load_from_file(str(expanded_path))
                    break

        # Override with environment variables
        config._load_from_env()

        return config

    @classmethod
    def _load_from_file(cls, file_path: str) -> 'CQEConfig':
        """Load configuration from YAML or JSON file"""

        path = Path(file_path)

        try:
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {path.suffix}")

            return cls(**data)

        except Exception as e:
            print(f"Warning: Could not load config from {file_path}: {e}")
            return cls()  # Return default config

    def _load_from_env(self):
        """Override configuration with environment variables"""

        env_mappings = {
            "CQE_LOG_LEVEL": ("log_level", str),
            "CQE_MAX_ATOMS": ("max_atoms", int),
            "CQE_VALIDATION_TIMEOUT": ("validation_timeout", float),
            "CQE_ENABLE_CACHING": ("enable_caching", lambda x: x.lower() == 'true'),
            "CQE_E8_PRECISION": ("e8_precision", float),
            "CQE_SLICE_TIMEOUT": ("slice_timeout", float),
            "CQE_PARITY_LANES": ("parity_lanes", int),
            "CQE_ENERGY_CONSTRAINT_STRICT": ("energy_constraint_strict", lambda x: x.lower() == 'true'),
            "CQE_MAX_CONCURRENT": ("max_concurrent_validations", int),
            "CQE_MEMORY_LIMIT_MB": ("memory_limit_mb", int),
            "CQE_CACHE_SIZE_MB": ("cache_size_mb", int),
            "CQE_LEDGER_PATH": ("ledger_path", str),
            "CQE_ATOM_STORAGE_PATH": ("atom_storage_path", str),
            "CQE_API_HOST": ("api_host", str),
            "CQE_API_PORT": ("api_port", int),
            "CQE_DEBUG_MODE": ("debug_mode", lambda x: x.lower() == 'true'),
            "CQE_ENABLE_METRICS": ("enable_metrics", lambda x: x.lower() == 'true'),
        }

        for env_var, (attr_name, converter) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = converter(os.environ[env_var])
                    setattr(self, attr_name, value)
                except Exception as e:
                    print(f"Warning: Could not parse {env_var}: {e}")

    def save(self, file_path: str):
        """Save configuration to file"""

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        config_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

        # Save as YAML by default
        with open(path, 'w') as f:
            if path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                yaml.safe_dump(config_dict, f, default_flow_style=False)

    def get_slice_config(self, slice_name: str) -> Dict[str, Any]:
        """Get configuration for a specific slice"""
        return {
            "timeout": self.slice_timeout,
            "enabled": slice_name not in self.disabled_slices,
            "debug": self.debug_mode
        }

    def validate(self) -> bool:
        """Validate configuration parameters"""

        errors = []

        # Validate numeric ranges
        if self.max_atoms <= 0:
            errors.append("max_atoms must be positive")

        if self.validation_timeout <= 0:
            errors.append("validation_timeout must be positive")

        if self.lattice_dimension != 8:
            errors.append("lattice_dimension must be 8 for E8 lattice")

        if self.parity_lanes not in [8, 16, 32, 64]:
            errors.append("parity_lanes must be 8, 16, 32, or 64")

        # Validate paths
        for path_attr in ["atom_storage_path"]:
            path_val = getattr(self, path_attr)
            try:
                Path(path_val).parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Invalid {path_attr}: {e}")

        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True

    def __str__(self):
        return f"CQEConfig(slices={37-len(self.disabled_slices)}/37, atoms≤{self.max_atoms})"
