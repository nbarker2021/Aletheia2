"""
Overlay caching system with Redis backend support
"""

from typing import Optional, Dict, List
from cqe.core.overlay import CQEOverlay
from cqe.storage.serialization import serialize_overlay, deserialize_overlay
import logging

logger = logging.getLogger(__name__)


class OverlayCache:
    """
    In-memory overlay cache with optional Redis backend.

    Provides:
    - Fast in-memory lookup
    - Persistence to Redis (if available)
    - LRU eviction policy
    - Statistics tracking
    """

    def __init__(self, max_size: int = 10000, redis_url: Optional[str] = None):
        """
        Initialize overlay cache.

        Args:
            max_size: Maximum number of overlays in memory
            redis_url: Optional Redis connection URL
        """
        self.max_size = max_size
        self._memory_cache: Dict[str, CQEOverlay] = {}
        self._access_order: List[str] = []

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'stores': 0
        }

        # Redis support (optional)
        self.redis_client = None
        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
                logger.info(f"Connected to Redis at {redis_url}")
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}. Using memory-only cache.")

    def get(self, overlay_id: str) -> Optional[CQEOverlay]:
        """
        Retrieve overlay from cache.

        Args:
            overlay_id: Overlay hash ID

        Returns:
            CQEOverlay if found, None otherwise
        """
        # Check memory cache first
        if overlay_id in self._memory_cache:
            self.stats['hits'] += 1
            self._update_access(overlay_id)
            return self._memory_cache[overlay_id]

        # Check Redis if available
        if self.redis_client:
            try:
                data = self.redis_client.get(f"cqe:overlay:{overlay_id}")
                if data:
                    overlay = deserialize_overlay(data.decode('utf-8'))
                    self._memory_cache[overlay_id] = overlay
                    self._update_access(overlay_id)
                    self.stats['hits'] += 1
                    return overlay
            except Exception as e:
                logger.error(f"Redis get error: {e}")

        self.stats['misses'] += 1
        return None

    def put(self, overlay: CQEOverlay) -> bool:
        """
        Store overlay in cache.

        Args:
            overlay: Overlay to store

        Returns:
            True if stored successfully
        """
        if not overlay.hash_id:
            logger.warning("Cannot cache overlay without hash_id")
            return False

        # Evict if necessary
        if len(self._memory_cache) >= self.max_size and overlay.hash_id not in self._memory_cache:
            self._evict_lru()

        # Store in memory
        self._memory_cache[overlay.hash_id] = overlay
        self._update_access(overlay.hash_id)
        self.stats['stores'] += 1

        # Store in Redis if available
        if self.redis_client:
            try:
                serialized = serialize_overlay(overlay)
                self.redis_client.set(
                    f"cqe:overlay:{overlay.hash_id}",
                    serialized,
                    ex=86400  # 24 hour TTL
                )
            except Exception as e:
                logger.error(f"Redis put error: {e}")

        return True

    def _update_access(self, overlay_id: str):
        """Update LRU access order"""
        if overlay_id in self._access_order:
            self._access_order.remove(overlay_id)
        self._access_order.append(overlay_id)

    def _evict_lru(self):
        """Evict least recently used overlay"""
        if self._access_order:
            lru_id = self._access_order.pop(0)
            if lru_id in self._memory_cache:
                del self._memory_cache[lru_id]
                self.stats['evictions'] += 1
                logger.debug(f"Evicted overlay {lru_id[:8]}")

    def size(self) -> int:
        """Return current cache size"""
        return len(self._memory_cache)

    def clear(self):
        """Clear all cached overlays"""
        self._memory_cache.clear()
        self._access_order.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0

        return {
            **self.stats,
            'size': len(self._memory_cache),
            'max_size': self.max_size,
            'hit_rate': hit_rate
        }
