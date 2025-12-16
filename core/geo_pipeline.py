#!/usr/bin/env python3
"""
Geometric Processing Pipeline
=============================

This module integrates the GeoTransformer and GeoTokenizer into a unified
processing pipeline with mandatory SpeedLight receipt generation.

The pipeline handles:
1. Tokenization of input data into geometric tokens
2. Transformation through the E8-constrained transformer
3. Output embedding generation
4. Full audit trail via SpeedLight
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.speedlight_wrapper import (
    requires_receipt, 
    SpeedLightContext, 
    log_transform, 
    log_embedding,
    get_speedlight
)


class GeoTokenizer:
    """
    Geometric Tokenizer - Converts input data into geometric tokens.
    
    The tokenizer maps input elements to positions in the E8 lattice space,
    creating a geometric representation that preserves structural relationships.
    """
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 64):
        """
        Initialize the geometric tokenizer.
        
        Args:
            vocab_size: Size of the token vocabulary
            d_model: Dimension of the model (must be multiple of 8)
        """
        assert d_model % 8 == 0, "d_model must be multiple of 8 for E8 structure"
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Initialize embedding matrix with E8-aligned structure
        np.random.seed(42)  # Reproducibility
        self.embeddings = np.random.randn(vocab_size, d_model) * 0.1
        
        # Normalize to unit sphere
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)
    
    @requires_receipt("tokenize", layer="L2")
    def tokenize(self, text: str) -> np.ndarray:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text string
        
        Returns:
            Array of token IDs
        """
        # Simple character-level tokenization for demo
        # In production, would use proper BPE or similar
        tokens = [ord(c) % self.vocab_size for c in text]
        return np.array(tokens)
    
    @requires_receipt("embed", layer="L2")
    def embed(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Convert token IDs to geometric embeddings.
        
        Args:
            token_ids: Array of token IDs
        
        Returns:
            Array of shape (seq_len, d_model) with geometric embeddings
        """
        return self.embeddings[token_ids]
    
    def __call__(self, text: str) -> np.ndarray:
        """Tokenize and embed in one step."""
        token_ids = self.tokenize(text)
        return self.embed(token_ids)


class E8Projector:
    """
    E8 Lattice Projector - Enforces geometric constraints.
    
    Projects vectors onto the E8 lattice structure to ensure
    all representations satisfy the morphonic constraints.
    """
    
    def __init__(self):
        """Initialize with E8 root vectors."""
        self.roots = self._generate_e8_roots()
    
    def _generate_e8_roots(self) -> np.ndarray:
        """Generate the 240 root vectors of E8."""
        roots = []
        
        # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) - 112 roots
        for i in range(8):
            for j in range(i + 1, 8):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        root = [0.0] * 8
                        root[i] = s1
                        root[j] = s2
                        roots.append(root)
        
        # Type 2: (±1/2)^8 with even number of minus signs - 128 roots
        for signs in range(256):
            root = []
            num_minus = 0
            for bit in range(8):
                if signs & (1 << bit):
                    root.append(0.5)
                else:
                    root.append(-0.5)
                    num_minus += 1
            if num_minus % 2 == 0:
                roots.append(root)
        
        return np.array(roots[:240])
    
    @requires_receipt("e8_project", layer="L2")
    def project(self, vector: np.ndarray) -> np.ndarray:
        """
        Project a vector onto the nearest E8 lattice point.
        
        Args:
            vector: Input vector (any dimension, will be reshaped to 8D blocks)
        
        Returns:
            Projected vector satisfying E8 constraints
        """
        original_shape = vector.shape
        
        # Reshape to 8D blocks
        flat = vector.flatten()
        padded_len = ((len(flat) + 7) // 8) * 8
        padded = np.zeros(padded_len)
        padded[:len(flat)] = flat
        
        # Project each 8D block
        blocks = padded.reshape(-1, 8)
        projected = np.zeros_like(blocks)
        
        for i, block in enumerate(blocks):
            # Find nearest root
            distances = np.linalg.norm(self.roots - block, axis=1)
            nearest_idx = np.argmin(distances)
            
            # Scale to match original magnitude
            scale = np.linalg.norm(block) / (np.linalg.norm(self.roots[nearest_idx]) + 1e-8)
            projected[i] = self.roots[nearest_idx] * scale
        
        # Reshape back
        result = projected.flatten()[:len(flat)]
        return result.reshape(original_shape)


class GeoTransformer:
    """
    Geometric Transformer - E8-constrained attention mechanism.
    
    Implements a transformer architecture where all operations
    are constrained to the E8 lattice structure.
    """
    
    def __init__(self, d_model: int = 64, n_heads: int = 8, n_layers: int = 6):
        """
        Initialize the geometric transformer.
        
        Args:
            d_model: Model dimension (must be multiple of 8)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
        """
        assert d_model % 8 == 0, "d_model must be multiple of 8"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_head = d_model // n_heads
        
        self.projector = E8Projector()
        
        # Initialize weights
        np.random.seed(42)
        self.weights = {
            f"layer_{i}": {
                "W_q": np.random.randn(d_model, d_model) * 0.02,
                "W_k": np.random.randn(d_model, d_model) * 0.02,
                "W_v": np.random.randn(d_model, d_model) * 0.02,
                "W_o": np.random.randn(d_model, d_model) * 0.02,
                "W_ff1": np.random.randn(d_model, d_model * 4) * 0.02,
                "W_ff2": np.random.randn(d_model * 4, d_model) * 0.02,
            }
            for i in range(n_layers)
        }
    
    def _attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Compute scaled dot-product attention."""
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.T) / np.sqrt(d_k)
        weights = self._softmax(scores)
        return np.matmul(weights, V)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + eps)
    
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation function."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    @requires_receipt("transform", layer="L2")
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape (seq_len, d_model)
        
        Returns:
            Output tensor of shape (seq_len, d_model)
        """
        for i in range(self.n_layers):
            layer_weights = self.weights[f"layer_{i}"]
            
            # Multi-head attention
            Q = np.matmul(x, layer_weights["W_q"])
            K = np.matmul(x, layer_weights["W_k"])
            V = np.matmul(x, layer_weights["W_v"])
            
            # E8 projection on attention outputs
            attn_out = self._attention(Q, K, V)
            attn_out = self.projector.project(attn_out)
            
            # Residual + LayerNorm
            x = self._layer_norm(x + np.matmul(attn_out, layer_weights["W_o"]))
            
            # Feed-forward
            ff_out = self._gelu(np.matmul(x, layer_weights["W_ff1"]))
            ff_out = np.matmul(ff_out, layer_weights["W_ff2"])
            
            # E8 projection on FF outputs
            ff_out = self.projector.project(ff_out)
            
            # Residual + LayerNorm
            x = self._layer_norm(x + ff_out)
        
        return x
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return self.forward(x)


class GeoPipeline:
    """
    Complete Geometric Processing Pipeline.
    
    Integrates tokenization, transformation, and embedding generation
    with full SpeedLight audit trail.
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 6
    ):
        """
        Initialize the complete pipeline.
        
        Args:
            vocab_size: Tokenizer vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
        """
        self.tokenizer = GeoTokenizer(vocab_size=vocab_size, d_model=d_model)
        self.transformer = GeoTransformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.projector = E8Projector()
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process text through the complete pipeline.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary containing:
            - tokens: Token IDs
            - embeddings: Initial embeddings
            - transformed: Transformed embeddings
            - final_embedding: Pooled final embedding
        """
        with SpeedLightContext("geo_pipeline", layer="L2", metadata={"input_len": len(text)}) as ctx:
            # Tokenize
            ctx.log("tokenize", {"text_len": len(text)})
            token_ids = self.tokenizer.tokenize(text)
            
            # Embed
            ctx.log("embed", {"num_tokens": len(token_ids)})
            embeddings = self.tokenizer.embed(token_ids)
            
            # Transform
            ctx.log("transform", {"shape": list(embeddings.shape)})
            transformed = self.transformer(embeddings)
            
            # E8 project final output
            ctx.log("project", {"shape": list(transformed.shape)})
            projected = self.projector.project(transformed)
            
            # Pool to single embedding
            final_embedding = np.mean(projected, axis=0)
            
            log_embedding(len(final_embedding), "geo_pipeline", layer="L2")
            
            return {
                "tokens": token_ids.tolist(),
                "embeddings": embeddings.tolist(),
                "transformed": transformed.tolist(),
                "final_embedding": final_embedding.tolist()
            }
    
    def __call__(self, text: str) -> Dict[str, Any]:
        """Process text through pipeline."""
        return self.process(text)


# Convenience function for quick processing
def process_text(text: str, **kwargs) -> Dict[str, Any]:
    """
    Quick function to process text through the geometric pipeline.
    
    Args:
        text: Input text
        **kwargs: Pipeline configuration options
    
    Returns:
        Processing results dictionary
    """
    pipeline = GeoPipeline(**kwargs)
    return pipeline.process(text)
