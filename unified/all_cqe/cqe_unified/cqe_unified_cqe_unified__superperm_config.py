
from typing import Any, Dict

class ConfigManager:
    def __init__(self, config_file: str = 'config.json'):
        self.settings: Dict[str, Any] = {
            'n': 7,
            'auto_loop': False,
            'strategy': 'bouncing_batch',
            'evaluation_metric': 'comprehensive',
            'length_weight': 1.0,
            'imperfection_weight': 10000000.0,
            'winner_loser_weight': 4.5,
            'layout_memory_weight': 0.35,
            'imbalance_weight': 0.02,
            'connectivity_weight': 1.4,
            'symmetry_weight': 0.0,
            'extensibility_weight': 2.0,
            'grid_dimensions': [3,3,3],
            'bouncing_batch_size': 7,
            'bouncing_batch_iterations': 25,
            'store_full_permutations': False,
            'k_mer_size': 6,
            'data_file': 'superperm_data.json',
            'strategy_thresholds': {'small':5, 'medium':7},
            'auto_adjust': False,
            'auto_adjust_params': {}
        }
