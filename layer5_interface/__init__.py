"""
Layer 5: Interface & Applications

Provides user-facing interfaces and integration points:
- Native SDK for geometric operations
- Standard bridge for traditional APIs
- Integration with applications
"""

from .sdk import CQESDK, CQEResult

__all__ = ['CQESDK', 'CQEResult']
