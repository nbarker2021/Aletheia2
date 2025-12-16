#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monster Moonshine Database
===========================
SQLite database with Monster group j-invariant features for geometric embeddings.

Features:
- moonshine_feature(dim=32) - 32D vector from Monster group representations
- radial_angle_hist() - Geometric histogram features
- summarize_lane() - CQE channel features
- fuse() - Combine multiple feature spaces

Database Schema:
- items: vector storage with metadata
- charts: feature space definitions (moonshine, geom, cqe)
- item_charts: many-to-many relationship
- logs: audit trail
"""

import hashlib
import json
import math
import os
import sqlite3
import time
from typing import List, Tuple, Dict, Any, Optional

# ───────────────────────────── Monster Moonshine Features ────────────────────

# j-invariant coefficients (first few terms of the q-expansion)
J_COEFFS = [
    1, 744, 196884, 21493760, 864299970, 20245856256,
    333202640600, 4252023300096, 44656994071935, 401490886656000
]

# Monstrous moonshine character tables (first few dimensions)
# These are the dimensions of irreducible representations of the Monster group
MT_1A = [1, 196883, 21296876, 842609326, 18538750076, 19360062527, 293553734298]
MT_2A = [1, 4371, 96256, 1240002, 8503056, 36288252, 108839451]
MT_3A = [1, 782, 10773, 80652, 301784, 749808, 1506943]

def moonshine_feature(dim: int = 32) -> List[float]:
    """
    Generate a moonshine feature vector using Monster group representations.
    
    The Monster group M has order ~8×10^53 and is connected to the j-invariant
    via monstrous moonshine. The first coefficient 196884 = 196883 + 1 where
    196883 is the dimension of the smallest nontrivial irrep.
    
    Args:
        dim: Dimension of output vector (default 32)
    
    Returns:
        List of floats representing Monster group features
    """
    feat = []
    
    # Use j-invariant coefficients
    for i, c in enumerate(J_COEFFS[:min(10, dim//3)]):
        # Normalize by log to handle huge numbers
        feat.append(math.log(1 + abs(c)))
    
    # Use character table dimensions (1A conjugacy class)
    for i, d in enumerate(MT_1A[:min(7, dim//4)]):
        feat.append(math.log(1 + d))
    
    # Use 2A conjugacy class
    for i, d in enumerate(MT_2A[:min(7, dim//4)]):
        feat.append(math.log(1 + d))
    
    # Use 3A conjugacy class
    for i, d in enumerate(MT_3A[:min(7, dim//4)]):
        feat.append(math.log(1 + d))
    
    # Pad or truncate to exact dimension
    while len(feat) < dim:
        feat.append(0.0)
    
    return feat[:dim]

def radial_angle_hist(points: List[Tuple[float, float]], rbins: int = 16, abins: int = 16) -> List[float]:
    """
    Create radial and angular histogram features from 2D points.
    
    Args:
        points: List of (x, y) coordinates
        rbins: Number of radial bins
        abins: Number of angular bins
    
    Returns:
        Concatenated histogram features
    """
    if not points:
        return [0.0] * (rbins + abins)
    
    # Compute centroid
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    
    # Compute radii and angles relative to centroid
    radii = []
    angles = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        r = math.sqrt(dx*dx + dy*dy)
        theta = math.atan2(dy, dx) % (2 * math.pi)
        radii.append(r)
        angles.append(theta)
    
    # Create histograms
    max_r = max(radii) if radii else 1.0
    r_hist = [0] * rbins
    a_hist = [0] * abins
    
    for r, a in zip(radii, angles):
        r_idx = min(rbins - 1, int(rbins * (r / max_r)))
        a_idx = min(abins - 1, int(abins * (a / (2 * math.pi))))
        r_hist[r_idx] += 1
        a_hist[a_idx] += 1
    
    # Normalize
    total = len(points)
    r_hist = [x / total for x in r_hist]
    a_hist = [x / total for x in a_hist]
    
    return r_hist + a_hist

def summarize_lane(channel_data: List[float], bins: int = 8) -> List[float]:
    """
    Summarize CQE channel/lane data into histogram features.
    
    Args:
        channel_data: List of channel values
        bins: Number of bins for histogram
    
    Returns:
        Histogram features
    """
    if not channel_data:
        return [0.0] * bins
    
    min_val = min(channel_data)
    max_val = max(channel_data)
    range_val = max_val - min_val if max_val > min_val else 1.0
    
    hist = [0] * bins
    for val in channel_data:
        idx = min(bins - 1, int(bins * ((val - min_val) / range_val)))
        hist[idx] += 1
    
    # Normalize
    total = len(channel_data)
    return [x / total for x in hist]

def fuse(features: List[List[float]]) -> List[float]:
    """
    Fuse multiple feature vectors by concatenation.
    
    Args:
        features: List of feature vectors
    
    Returns:
        Concatenated feature vector
    """
    result = []
    for feat in features:
        result.extend(feat)
    return result

# ───────────────────────────── Database ──────────────────────────────────────

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)

class MonsterMoonshinDB:
    """
    SQLite database with Monster Moonshine features.
    
    Schema:
    - items: id, vector (JSON), metadata (JSON), created_at
    - charts: id, name, description
    - item_charts: item_id, chart_id, feature_vector (JSON)
    - logs: id, timestamp, operation, details (JSON)
    """
    
    def __init__(self, path: str = "./data/monster_moonshine.db"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vector TEXT NOT NULL,
                metadata TEXT,
                created_at REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS charts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT
            );
            
            CREATE TABLE IF NOT EXISTS item_charts (
                item_id INTEGER NOT NULL,
                chart_id INTEGER NOT NULL,
                feature_vector TEXT NOT NULL,
                PRIMARY KEY (item_id, chart_id),
                FOREIGN KEY (item_id) REFERENCES items(id),
                FOREIGN KEY (chart_id) REFERENCES charts(id)
            );
            
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                operation TEXT NOT NULL,
                details TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_items_created ON items(created_at);
            CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
        """)
        self.conn.commit()
        
        # Ensure default charts exist
        for name, desc in [
            ("moonshine", "Monster Moonshine j-invariant features"),
            ("geom", "Geometric radial/angular histogram features"),
            ("cqe", "CQE channel/lane summary features")
        ]:
            self.conn.execute(
                "INSERT OR IGNORE INTO charts (name, description) VALUES (?, ?)",
                (name, desc)
            )
        self.conn.commit()
    
    def add_item(self, vector: List[float], metadata: Optional[Dict[str, Any]] = None,
                 chart_features: Optional[Dict[str, List[float]]] = None) -> int:
        """
        Add an item to the database.
        
        Args:
            vector: Main vector to store
            metadata: Optional metadata dictionary
            chart_features: Optional dict of chart_name -> feature_vector
        
        Returns:
            Item ID
        """
        cursor = self.conn.execute(
            "INSERT INTO items (vector, metadata, created_at) VALUES (?, ?, ?)",
            (json.dumps(vector), json.dumps(metadata or {}), time.time())
        )
        item_id = cursor.lastrowid
        
        # Add chart features if provided
        if chart_features:
            for chart_name, feat_vec in chart_features.items():
                chart_id = self.conn.execute(
                    "SELECT id FROM charts WHERE name = ?", (chart_name,)
                ).fetchone()
                if chart_id:
                    self.conn.execute(
                        "INSERT INTO item_charts (item_id, chart_id, feature_vector) VALUES (?, ?, ?)",
                        (item_id, chart_id[0], json.dumps(feat_vec))
                    )
        
        self.conn.commit()
        self._log("add_item", {"item_id": item_id, "vector_dim": len(vector)})
        return item_id
    
    def get_item(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get an item by ID."""
        row = self.conn.execute(
            "SELECT id, vector, metadata, created_at FROM items WHERE id = ?",
            (item_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return {
            "id": row[0],
            "vector": json.loads(row[1]),
            "metadata": json.loads(row[2]),
            "created_at": row[3]
        }
    
    def list_items(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List items with pagination."""
        rows = self.conn.execute(
            "SELECT id, vector, metadata, created_at FROM items ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        
        return [
            {
                "id": row[0],
                "vector": json.loads(row[1]),
                "metadata": json.loads(row[2]),
                "created_at": row[3]
            }
            for row in rows
        ]
    
    def search(self, query_vector: List[float], chart_name: Optional[str] = None,
               top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar items using cosine similarity.
        
        Args:
            query_vector: Query vector
            chart_name: Optional chart name to search within
            top_k: Number of results to return
        
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if chart_name:
            # Search within a specific chart
            chart_id = self.conn.execute(
                "SELECT id FROM charts WHERE name = ?", (chart_name,)
            ).fetchone()
            if not chart_id:
                return []
            
            rows = self.conn.execute(
                "SELECT item_id, feature_vector FROM item_charts WHERE chart_id = ?",
                (chart_id[0],)
            ).fetchall()
            
            results = []
            for item_id, feat_json in rows:
                feat_vec = json.loads(feat_json)
                sim = cosine_similarity(query_vector, feat_vec)
                results.append((item_id, sim))
        else:
            # Search main vectors
            rows = self.conn.execute("SELECT id, vector FROM items").fetchall()
            results = []
            for item_id, vec_json in rows:
                vec = json.loads(vec_json)
                sim = cosine_similarity(query_vector, vec)
                results.append((item_id, sim))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        item_count = self.conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        chart_count = self.conn.execute("SELECT COUNT(*) FROM charts").fetchone()[0]
        log_count = self.conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        
        return {
            "items": item_count,
            "charts": chart_count,
            "logs": log_count
        }
    
    def _log(self, operation: str, details: Dict[str, Any]):
        """Log an operation."""
        self.conn.execute(
            "INSERT INTO logs (timestamp, operation, details) VALUES (?, ?, ?)",
            (time.time(), operation, json.dumps(details))
        )
        self.conn.commit()
    
    def close(self):
        """Close the database connection."""
        self.conn.close()
