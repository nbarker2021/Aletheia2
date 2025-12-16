"""
Base adapter interface for domain-specific feature extraction
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class DomainAdapter(ABC):
    """
    Abstract base class for domain adapters.

    All adapters must extract 8-dimensional feature vectors
    from domain-specific content for E8 embedding.
    """

    @abstractmethod
    def extract_features(self, content: Any) -> np.ndarray:
        """
        Extract 8-dimensional feature vector from content.

        Args:
            content: Domain-specific content

        Returns:
            8-dimensional numpy array of features
        """
        pass

    @abstractmethod
    def validate_content(self, content: Any) -> bool:
        """
        Validate content is appropriate for this adapter.

        Args:
            content: Content to validate

        Returns:
            True if valid
        """
        pass

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to unit sphere or standard range.

        Args:
            features: Raw features

        Returns:
            Normalized features
        """
        # Z-score normalization
        mean = np.mean(features)
        std = np.std(features)

        if std > 1e-8:
            return (features - mean) / std
        return features - mean
