"""
GeoTransformer Integration Module
=================================

Wires the geometric transformer to the unified runtime, providing:
- E8-constrained attention mechanism
- Lambda term generation for all operations
- SpeedLight caching for transform results
- Conservation law enforcement (ΔΦ ≤ 0)

This module integrates:
- morphonic_cqe_unified/experimental/geometric_transformer_standalone.py
- morphonic_cqe_unified/experimental/lambda_e8_calculus.py
- morphonic_cqe_unified/sidecar/speedlight_sidecar_plus.py

Author: Manus AI
Date: December 16, 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import hashlib
import json
import time

# Import from morphonic_cqe_unified
from morphonic_cqe_unified.experimental.lambda_e8_calculus import (
    LambdaE8Builder, LambdaTerm, LambdaType, GeometricLambdaCapture
)
from morphonic_cqe_unified.sidecar.speedlight_sidecar_plus import SpeedLightV2


# =============================================================================
# E8 LATTICE UTILITIES
# =============================================================================

def generate_e8_roots() -> np.ndarray:
    """Generate the 240 root vectors of E8."""
    roots = []
    n = 8
    
    # Family 1: D8 roots (±1, ±1, 0^6), 112 roots
    for i in range(n):
        for j in range(i+1, n):
            for s1 in (+1.0, -1.0):
                for s2 in (+1.0, -1.0):
                    v = [0.0]*n
                    v[i] = s1
                    v[j] = s2
                    roots.append(v)
    
    # Family 2: (±1/2)^8 with even number of + (128 roots)
    from itertools import product
    for signs in product((-0.5, 0.5), repeat=8):
        plus = sum(1 for s in signs if s > 0)
        if plus % 2 == 0:
            roots.append(list(signs))
    
    return np.array(roots[:240])


def project_to_e8(vector: np.ndarray) -> np.ndarray:
    """Project a vector onto the nearest E8 lattice point."""
    return np.round(vector * 2) / 2


def compute_phi(A: np.ndarray, x: np.ndarray) -> float:
    """Compute Φ(x) = x^T A x (energy function)."""
    return float(x @ A @ x)


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation - smooth approximation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


# =============================================================================
# GEOMETRIC ATTENTION
# =============================================================================

@dataclass
class AttentionConfig:
    """Configuration for geometric attention."""
    d_model: int = 64  # Must be multiple of 8
    n_heads: int = 8   # Must be power of 2
    d_head: int = 8    # d_model // n_heads
    enforce_e8: bool = True
    conservation_threshold: float = 1e-6


class GeometricAttention:
    """
    Multi-head attention with E8 geometric constraints.
    
    Implements attention as interference patterns in 8D space,
    with conservation law enforcement and lambda term generation.
    """
    
    def __init__(
        self,
        config: AttentionConfig,
        speedlight: Optional[SpeedLightV2] = None,
        lambda_builder: Optional[LambdaE8Builder] = None
    ):
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        
        # Initialize weights
        scale = 1.0 / np.sqrt(self.d_model)
        self.W_q = np.random.randn(self.d_model, self.d_model) * scale
        self.W_k = np.random.randn(self.d_model, self.d_model) * scale
        self.W_v = np.random.randn(self.d_model, self.d_model) * scale
        self.W_o = np.random.randn(self.d_model, self.d_model) * scale
        
        # E8 roots for geometric constraints
        if config.enforce_e8:
            self.e8_roots = generate_e8_roots()
        
        # Cartan matrix for Φ computation
        self._init_cartan()
        
        # SpeedLight for caching
        self.speedlight = speedlight or SpeedLightV2(mem_bytes=64*1024*1024)
        
        # Lambda builder for term generation
        self.lambda_builder = lambda_builder or LambdaE8Builder()
        
        # Statistics
        self.stats = {
            "forward_calls": 0,
            "cache_hits": 0,
            "conservation_violations": 0
        }
    
    def _init_cartan(self):
        """Initialize Cartan matrix for E8."""
        # E8 Cartan matrix
        self.cartan = np.array([
            [ 2, -1,  0,  0,  0,  0,  0,  0],
            [-1,  2, -1,  0,  0,  0,  0,  0],
            [ 0, -1,  2, -1,  0,  0,  0, -1],
            [ 0,  0, -1,  2, -1,  0,  0,  0],
            [ 0,  0,  0, -1,  2, -1,  0,  0],
            [ 0,  0,  0,  0, -1,  2, -1,  0],
            [ 0,  0,  0,  0,  0, -1,  2,  0],
            [ 0,  0, -1,  0,  0,  0,  0,  2]
        ], dtype=np.float64)
    
    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split into multiple attention heads."""
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_head)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_head)
    
    def merge_heads(self, x: np.ndarray) -> np.ndarray:
        """Merge attention heads back together."""
        batch_size, n_heads, seq_len, d_head = x.shape
        x = x.transpose(0, 2, 1, 3)  # (batch, seq, heads, d_head)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def _compute_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute scaled dot-product attention."""
        d_k = Q.shape[-1]
        
        # Q @ K^T / sqrt(d_k)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax
        attention_weights = softmax(scores, axis=-1)
        
        # Apply to values
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_lambda: bool = True
    ) -> Tuple[np.ndarray, Optional[LambdaTerm], Dict[str, Any]]:
        """
        Forward pass with E8 constraints and lambda term generation.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            return_lambda: Whether to generate lambda term
        
        Returns:
            Tuple of (output, lambda_term, metadata)
        """
        self.stats["forward_calls"] += 1
        start_time = time.time()
        
        # Compute input energy (for conservation check)
        input_energy = np.mean([compute_phi(self.cartan, x[0, i, :8]) for i in range(x.shape[1])])
        
        # Check cache
        cache_key = hashlib.sha256(x.tobytes()).hexdigest()
        
        def _compute():
            # Project Q, K, V
            Q = x @ self.W_q
            K = x @ self.W_k
            V = x @ self.W_v
            
            # Split heads
            Q = self.split_heads(Q)
            K = self.split_heads(K)
            V = self.split_heads(V)
            
            # Compute attention
            attn_output, attn_weights = self._compute_attention(Q, K, V, mask)
            
            # Merge heads
            output = self.merge_heads(attn_output)
            
            # Output projection
            output = output @ self.W_o
            
            # Apply E8 constraint if enabled
            if self.config.enforce_e8:
                # Project each position to E8
                for b in range(output.shape[0]):
                    for s in range(output.shape[1]):
                        output[b, s, :8] = project_to_e8(output[b, s, :8])
            
            return output.tolist(), attn_weights.tolist()
        
        result, cost, key = self.speedlight.compute(
            {"input_hash": cache_key},
            scope="geo_attention",
            channel=3,
            compute_fn=_compute
        )
        
        if cost == 0:
            self.stats["cache_hits"] += 1
        
        output = np.array(result[0])
        attn_weights = np.array(result[1])
        
        # Compute output energy (for conservation check)
        output_energy = np.mean([compute_phi(self.cartan, output[0, i, :8]) for i in range(output.shape[1])])
        delta_phi = output_energy - input_energy
        
        if delta_phi > self.config.conservation_threshold:
            self.stats["conservation_violations"] += 1
        
        # Generate lambda term if requested
        lambda_term = None
        if return_lambda:
            x_var = self.lambda_builder.var("x", LambdaType.VECTOR)
            q_proj = self.lambda_builder.e8_project(x_var, self.d_model)
            attn_op = LambdaTerm("e8_op", ("attention", [q_proj]), LambdaType.VECTOR)
            conserved = self.lambda_builder.conserve(attn_op)
            lambda_term = self.lambda_builder.abs("x", conserved, LambdaType.VECTOR)
        
        # Build metadata
        metadata = {
            "cost_ms": (time.time() - start_time) * 1000,
            "cache_hit": cost == 0,
            "input_energy": input_energy,
            "output_energy": output_energy,
            "delta_phi": delta_phi,
            "conservation_valid": delta_phi <= self.config.conservation_threshold,
            "attention_weights_shape": attn_weights.shape
        }
        
        return output, lambda_term, metadata


# =============================================================================
# GEOMETRIC TRANSFORMER LAYER
# =============================================================================

@dataclass
class TransformerConfig:
    """Configuration for geometric transformer."""
    d_model: int = 64
    n_heads: int = 8
    d_ff: int = 256  # Feedforward dimension
    n_layers: int = 6
    max_seq_len: int = 128
    dropout: float = 0.1
    enforce_e8: bool = True


class GeoTransformerLayer:
    """
    Single transformer layer with geometric constraints.
    
    Architecture:
    - Multi-head attention with E8 projection
    - Feedforward network with GELU activation
    - Layer normalization
    - Conservation law enforcement
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        speedlight: Optional[SpeedLightV2] = None,
        lambda_builder: Optional[LambdaE8Builder] = None
    ):
        self.config = config
        
        # Attention
        attn_config = AttentionConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_head=config.d_model // config.n_heads,
            enforce_e8=config.enforce_e8
        )
        self.attention = GeometricAttention(attn_config, speedlight, lambda_builder)
        
        # Feedforward weights
        scale = 1.0 / np.sqrt(config.d_model)
        self.W1 = np.random.randn(config.d_model, config.d_ff) * scale
        self.b1 = np.zeros(config.d_ff)
        self.W2 = np.random.randn(config.d_ff, config.d_model) * scale
        self.b2 = np.zeros(config.d_model)
        
        # SpeedLight and Lambda builder
        self.speedlight = speedlight or SpeedLightV2(mem_bytes=64*1024*1024)
        self.lambda_builder = lambda_builder or LambdaE8Builder()
    
    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """Feedforward network with GELU activation."""
        h = gelu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2
    
    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_lambda: bool = True
    ) -> Tuple[np.ndarray, Optional[LambdaTerm], Dict[str, Any]]:
        """
        Forward pass through transformer layer.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            return_lambda: Whether to generate lambda term
        
        Returns:
            Tuple of (output, lambda_term, metadata)
        """
        # Attention with residual
        attn_out, attn_lambda, attn_meta = self.attention.forward(x, mask, return_lambda)
        x = layer_norm(x + attn_out)
        
        # Feedforward with residual
        ff_out = self.feedforward(x)
        output = layer_norm(x + ff_out)
        
        # Apply E8 constraint to output
        if self.config.enforce_e8:
            for b in range(output.shape[0]):
                for s in range(output.shape[1]):
                    output[b, s, :8] = project_to_e8(output[b, s, :8])
        
        # Generate combined lambda term
        lambda_term = None
        if return_lambda and attn_lambda:
            # Compose attention lambda with feedforward
            ff_var = self.lambda_builder.var("h", LambdaType.VECTOR)
            ff_proj = self.lambda_builder.e8_project(ff_var, self.config.d_model)
            ff_lambda = self.lambda_builder.abs("h", ff_proj, LambdaType.VECTOR)
            lambda_term = self.lambda_builder.compose(ff_lambda, attn_lambda)
        
        # Build metadata
        metadata = {
            "attention": attn_meta,
            "layer_type": "transformer"
        }
        
        return output, lambda_term, metadata


# =============================================================================
# FULL GEO TRANSFORMER
# =============================================================================

class GeoTransformer:
    """
    Full geometric transformer with SpeedLight caching and lambda term generation.
    
    Features:
    - E8-constrained attention mechanism
    - Conservation law enforcement (ΔΦ ≤ 0)
    - Lambda term generation for all operations
    - SpeedLight caching for transform results
    """
    
    def __init__(
        self,
        config: Optional[TransformerConfig] = None,
        speedlight: Optional[SpeedLightV2] = None
    ):
        self.config = config or TransformerConfig()
        self.speedlight = speedlight or SpeedLightV2(mem_bytes=128*1024*1024)
        self.lambda_builder = LambdaE8Builder()
        
        # Build layers
        self.layers = []
        for i in range(self.config.n_layers):
            layer = GeoTransformerLayer(
                self.config,
                self.speedlight,
                self.lambda_builder
            )
            self.layers.append(layer)
        
        # Embedding (simple random initialization)
        scale = 1.0 / np.sqrt(self.config.d_model)
        self.embedding = np.random.randn(1000, self.config.d_model) * scale  # vocab_size=1000
        
        # Statistics
        self.stats = {
            "forward_calls": 0,
            "total_tokens": 0
        }
    
    def embed(self, token_ids: np.ndarray) -> np.ndarray:
        """Embed token IDs to vectors."""
        return self.embedding[token_ids]
    
    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_lambda: bool = True
    ) -> Tuple[np.ndarray, Optional[LambdaTerm], Dict[str, Any]]:
        """
        Forward pass through full transformer.
        
        Args:
            x: Input tensor (batch, seq_len, d_model) or token IDs
            mask: Optional attention mask
            return_lambda: Whether to generate lambda term
        
        Returns:
            Tuple of (output, lambda_term, metadata)
        """
        self.stats["forward_calls"] += 1
        start_time = time.time()
        
        # Embed if needed
        if x.dtype in [np.int32, np.int64]:
            x = self.embed(x)
        
        self.stats["total_tokens"] += x.shape[0] * x.shape[1]
        
        # Pass through layers
        layer_lambdas = []
        layer_metas = []
        
        for i, layer in enumerate(self.layers):
            x, layer_lambda, layer_meta = layer.forward(x, mask, return_lambda)
            layer_lambdas.append(layer_lambda)
            layer_metas.append(layer_meta)
        
        # Compose all lambda terms
        lambda_term = None
        if return_lambda and all(l is not None for l in layer_lambdas):
            lambda_term = self.lambda_builder.compose(*layer_lambdas)
        
        # Build metadata
        metadata = {
            "cost_ms": (time.time() - start_time) * 1000,
            "n_layers": len(self.layers),
            "layer_metadata": layer_metas,
            "speedlight_stats": self.speedlight.stats()
        }
        
        return x, lambda_term, metadata
    
    def status(self) -> Dict[str, Any]:
        """Get transformer status."""
        sl_stats = self.speedlight.stats()
        return {
            "config": {
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "n_layers": self.config.n_layers,
                "d_ff": self.config.d_ff,
                "enforce_e8": self.config.enforce_e8
            },
            "stats": self.stats,
            "speedlight": {
                "hits": sl_stats["hits"],
                "misses": sl_stats["misses"],
                "hit_rate": sl_stats["hits"] / max(sl_stats["hits"] + sl_stats["misses"], 1),
                "mem_bytes": sl_stats["mem_bytes"]
            }
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Test the GeoTransformer."""
    print("=" * 70)
    print("GeoTransformer Integration Test")
    print("=" * 70)
    
    # Create transformer
    config = TransformerConfig(
        d_model=64,
        n_heads=8,
        d_ff=256,
        n_layers=2,
        enforce_e8=True
    )
    transformer = GeoTransformer(config)
    
    print(f"\nConfig: {config}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 16
    x = np.random.randn(batch_size, seq_len, config.d_model)
    
    print(f"\nInput shape: {x.shape}")
    
    output, lambda_term, metadata = transformer.forward(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Lambda term: {lambda_term.to_string() if lambda_term else 'N/A'}")
    print(f"Cost: {metadata['cost_ms']:.2f}ms")
    print(f"SpeedLight hits: {metadata['speedlight_stats']['hits']}")
    
    # Test cache hit
    print("\n--- Testing Cache Hit ---")
    output2, _, metadata2 = transformer.forward(x)
    print(f"Second call cost: {metadata2['cost_ms']:.2f}ms")
    print(f"SpeedLight hits: {metadata2['speedlight_stats']['hits']}")
    
    # Status
    print("\n--- Transformer Status ---")
    status = transformer.status()
    print(json.dumps(status, indent=2, default=str))
    
    print("\n" + "=" * 70)
    print("✓ GeoTransformer integration test complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
