"""
Glyph Lambda - Symbolic Operators in Geometric Space

Based on "The Universal Glyph Dictionary: Symbolic Operators in Geometric Space"

Every symbol can be embedded in E8 space and used as a computational operator.
Glyphs achieve 5-15x token compression while maintaining formal rigor.
"""

import numpy as np
import hashlib
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field


@dataclass
class Glyph:
    """A symbolic operator with E8 embedding."""
    symbol: str
    name: str
    e8_coords: np.ndarray
    digital_root: int
    operation: Callable
    category: str
    description: str = ""
    
    def __post_init__(self):
        if isinstance(self.e8_coords, list):
            self.e8_coords = np.array(self.e8_coords)
    
    def apply(self, *args) -> Any:
        """Apply this glyph operator."""
        return self.operation(*args)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "e8_coords": self.e8_coords.tolist(),
            "digital_root": self.digital_root,
            "category": self.category,
            "description": self.description
        }


def digital_root(n: int) -> int:
    """Compute digital root (repeated digit sum until single digit)."""
    while n > 9:
        n = sum(int(d) for d in str(abs(n)))
    return n


class GlyphRegistry:
    """Registry of all available glyph operators."""
    
    def __init__(self):
        self.glyphs: Dict[str, Glyph] = {}
        self._register_core_glyphs()
    
    def register(self, glyph: Glyph):
        """Register a glyph operator."""
        self.glyphs[glyph.symbol] = glyph
    
    def get(self, symbol: str) -> Optional[Glyph]:
        """Get a glyph by symbol."""
        return self.glyphs.get(symbol)
    
    def apply(self, symbol: str, *args) -> Any:
        """Apply a glyph operator by symbol."""
        glyph = self.get(symbol)
        if glyph is None:
            raise ValueError(f"Unknown glyph: {symbol}")
        return glyph.apply(*args)
    
    def list_by_category(self, category: str) -> List[Glyph]:
        """List all glyphs in a category."""
        return [g for g in self.glyphs.values() if g.category == category]
    
    def _register_core_glyphs(self):
        """Register the core CQE glyph operators."""
        
        # Quantifiers
        self.register(Glyph(
            symbol="∀",
            name="Universal Quantifier",
            e8_coords=[1, 1, 1, 1, 1, 1, 1, 1],
            digital_root=8,
            operation=lambda P, domain: all(P(x) for x in domain),
            category="QUANTIFIER",
            description="For all"
        ))
        
        self.register(Glyph(
            symbol="∃",
            name="Existential Quantifier",
            e8_coords=[1, 0, 0, 0, 0, 0, 0, 0],
            digital_root=1,
            operation=lambda P, domain: any(P(x) for x in domain),
            category="QUANTIFIER",
            description="There exists"
        ))
        
        # Geometric operators
        self.register(Glyph(
            symbol="⊕",
            name="Geometric Addition",
            e8_coords=[1, 1, 0, 0, 0, 0, 0, 0],
            digital_root=2,
            operation=lambda a, b: np.array(a) + np.array(b),
            category="GEOMETRIC",
            description="Vector addition"
        ))
        
        self.register(Glyph(
            symbol="⊖",
            name="Geometric Subtraction",
            e8_coords=[1, -1, 0, 0, 0, 0, 0, 0],
            digital_root=0,
            operation=lambda a, b: np.array(a) - np.array(b),
            category="GEOMETRIC",
            description="Vector subtraction"
        ))
        
        self.register(Glyph(
            symbol="⊗",
            name="Tensor Product",
            e8_coords=[1, 0, 1, 0, 1, 0, 1, 0],
            digital_root=4,
            operation=lambda a, b: np.outer(a, b),
            category="GEOMETRIC",
            description="Outer product"
        ))
        
        self.register(Glyph(
            symbol="⊙",
            name="Inner Product",
            e8_coords=[1, 1, 1, 1, 1, 1, 1, 1],
            digital_root=8,
            operation=lambda a, b: np.dot(a, b),
            category="GEOMETRIC",
            description="Dot product"
        ))
        
        self.register(Glyph(
            symbol="⥁",
            name="Toroidal Rotation",
            e8_coords=[0, 1, 0, 1, 0, 1, 0, 1],
            digital_root=4,
            operation=self._toroidal_rotate,
            category="GEOMETRIC",
            description="Rotate on torus"
        ))
        
        self.register(Glyph(
            symbol="⇄",
            name="Parity Flip",
            e8_coords=[-1, -1, -1, -1, -1, -1, -1, -1],
            digital_root=0,
            operation=lambda v: -np.array(v),
            category="GEOMETRIC",
            description="Reflection through origin"
        ))
        
        self.register(Glyph(
            symbol="↑",
            name="Embed",
            e8_coords=[0, 0, 0, 0, 0, 0, 0, 1],
            digital_root=1,
            operation=self._embed,
            category="GEOMETRIC",
            description="Embed into E8 space"
        ))
        
        self.register(Glyph(
            symbol="↓",
            name="Project",
            e8_coords=[0, 0, 0, 0, 0, 0, 0, -1],
            digital_root=8,
            operation=lambda v: v[0] if len(v) > 0 else 0,
            category="GEOMETRIC",
            description="Project from E8 space"
        ))
        
        # CQE-specific operators
        self.register(Glyph(
            symbol="×",
            name="Scale Up",
            e8_coords=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            digital_root=8,
            operation=lambda p: [x + 0.10 for x in p],
            category="CQE",
            description="Scale phases up by 0.10"
        ))
        
        self.register(Glyph(
            symbol="÷",
            name="Midpoint",
            e8_coords=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            digital_root=4,
            operation=lambda p: [(p[i] + p[(i+1) % len(p)]) / 2 for i in range(len(p))],
            category="CQE",
            description="Midpoint between adjacent phases"
        ))
        
        self.register(Glyph(
            symbol="%",
            name="Modular Wrap",
            e8_coords=[1, 0, 0, 0, 0, 0, 0, 0],
            digital_root=1,
            operation=lambda p: [((x % 1.0) + 1.0) % 1.0 for x in p],
            category="CQE",
            description="Wrap phases to [0, 1)"
        ))
        
        self.register(Glyph(
            symbol="~",
            name="Parity Fix",
            e8_coords=[0, 0, 0, 0, 0, 0, 0, 0],
            digital_root=0,
            operation=self._parity_fix,
            category="CQE",
            description="Fix parity by flipping minimal entry"
        ))
        
        self.register(Glyph(
            symbol="#2",
            name="Kissing Adjust",
            e8_coords=[2, 0, 0, 0, 0, 0, 0, 0],
            digital_root=2,
            operation=self._kissing_adjust,
            category="CQE",
            description="Adjust to kissing number 2"
        ))
        
        # Logic operators
        self.register(Glyph(
            symbol="∧",
            name="And",
            e8_coords=[1, 1, 0, 0, 0, 0, 0, 0],
            digital_root=2,
            operation=lambda a, b: a and b,
            category="LOGIC",
            description="Logical conjunction"
        ))
        
        self.register(Glyph(
            symbol="∨",
            name="Or",
            e8_coords=[1, 0, 1, 0, 0, 0, 0, 0],
            digital_root=2,
            operation=lambda a, b: a or b,
            category="LOGIC",
            description="Logical disjunction"
        ))
        
        self.register(Glyph(
            symbol="¬",
            name="Not",
            e8_coords=[-1, 0, 0, 0, 0, 0, 0, 0],
            digital_root=8,
            operation=lambda a: not a,
            category="LOGIC",
            description="Logical negation"
        ))
        
        self.register(Glyph(
            symbol="⇒",
            name="Implies",
            e8_coords=[0, 1, 0, 0, 0, 0, 0, 0],
            digital_root=1,
            operation=lambda a, b: (not a) or b,
            category="LOGIC",
            description="Logical implication"
        ))
    
    def _toroidal_rotate(self, v: np.ndarray, theta: float) -> np.ndarray:
        """Rotate vector on torus."""
        v = np.array(v)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Rotate pairs of coordinates
        result = v.copy()
        for i in range(0, len(v) - 1, 2):
            result[i] = cos_t * v[i] - sin_t * v[i + 1]
            result[i + 1] = sin_t * v[i] + cos_t * v[i + 1]
        return result
    
    def _embed(self, x: float) -> np.ndarray:
        """Embed scalar into E8."""
        result = np.zeros(8)
        result[0] = x
        return result
    
    def _parity_fix(self, c: List[int]) -> List[int]:
        """Fix parity by flipping minimal entry."""
        c = c[:]
        if sum(c) % 2 == 1:
            i = min(range(len(c)), key=lambda i: c[i])
            c[i] = 1
        return c
    
    def _kissing_adjust(self, p: List[float]) -> List[float]:
        """Adjust to kissing number 2."""
        p = p[:]
        active = sum(1 for x in p if abs(x) > 1e-9)
        if active < 2:
            i = min(range(len(p)), key=lambda i: abs(p[i]))
            p[i] += 0.05
        elif active > 2:
            i = max(range(len(p)), key=lambda i: abs(p[i]))
            p[i] *= 0.8
        return p


class GlyphState:
    """State for glyph calculus operations."""
    
    def __init__(self, phases: List[float], cartan: List[int]):
        self.phases = phases
        self.cartan = cartan
    
    def phi(self) -> float:
        """Compute Φ objective function."""
        p, c = self.phases, self.cartan
        geom = sum((p[i] - p[(i + 1) % len(p)]) ** 2 for i in range(len(p)))
        parity = sum(c) % 2
        spars = sum(c)
        active = sum(1 for x in p if abs(x) > 1e-9)
        kiss = abs(active - 2)
        return geom + 5 * parity + 0.5 * spars + 0.1 * kiss
    
    def parity_ok(self) -> bool:
        """Check if parity is even."""
        return sum(self.cartan) % 2 == 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phases": self.phases,
            "cartan": self.cartan,
            "phi": self.phi(),
            "parity_ok": self.parity_ok()
        }


class GlyphCalculus:
    """
    Glyph-based calculus for CQE operations.
    
    Provides token-efficient computation using symbolic operators.
    """
    
    def __init__(self):
        self.registry = GlyphRegistry()
    
    def apply_sequence(self, state: GlyphState, ops: List[str]) -> Tuple[GlyphState, Dict]:
        """Apply a sequence of glyph operators."""
        for op in ops:
            state = self.apply_op(state, op)
        
        return state, {"ops": ops, "final_phi": state.phi()}
    
    def apply_op(self, state: GlyphState, op: str) -> GlyphState:
        """Apply a single glyph operator to state."""
        p, c = state.phases[:], state.cartan[:]
        
        if op == "×":
            p = [x + 0.10 for x in p]
        elif op == "÷":
            p = [(p[i] + p[(i + 1) % len(p)]) / 2 for i in range(len(p))]
        elif op == "%":
            p = [((x % 1.0) + 1.0) % 1.0 for x in p]
        elif op == "~":
            if sum(c) % 2 == 1:
                i = min(range(len(c)), key=lambda i: c[i])
                c[i] = 1
        elif op == "#2":
            active = sum(1 for x in p if abs(x) > 1e-9)
            if active < 2:
                i = min(range(len(p)), key=lambda i: abs(p[i]))
                p[i] += 0.05
            elif active > 2:
                i = max(range(len(p)), key=lambda i: abs(p[i]))
                p[i] *= 0.8
        
        return GlyphState(p, c)
    
    def run_rung(
        self,
        state: GlyphState,
        rails: List[List[str]]
    ) -> Tuple[GlyphState, Dict]:
        """
        Run a rung of the ladder - try multiple operator sequences.
        
        Returns the best result that passes the gate.
        """
        trials = []
        
        for seq in rails:
            s1 = state
            for op in seq:
                s1 = self.apply_op(s1, op)
            trials.append((seq, s1))
        
        # Sort by phi (lower is better)
        trials.sort(key=lambda t: t[1].phi())
        winner_seq, s1 = trials[0]
        
        # Gate check: parity must be even and phi must not increase
        accepted = s1.parity_ok() and (s1.phi() <= state.phi())
        
        receipt = {
            "winner_seq": winner_seq,
            "phi_before": state.phi(),
            "phi_after": s1.phi(),
            "parity_even": s1.parity_ok(),
            "accepted": accepted
        }
        
        return (s1 if accepted else state), receipt
    
    def init_state(self, n: int = 8, seed: int = 0) -> GlyphState:
        """Initialize a random state."""
        import random
        random.seed(seed)
        p = [random.uniform(-1, 1) for _ in range(n)]
        c = [0] * n
        return GlyphState(p, c)


class OverlayRegistry:
    """Registry for glyph overlays (operator sequences)."""
    
    def __init__(self):
        self.db: Dict[str, Dict] = {}
    
    def set(self, glyph: str, ops: List[str], status: str = "PENDING"):
        """Register an overlay."""
        self.db[glyph] = {"ops": ops, "status": status, "evidence": []}
    
    def promote(self, glyph: str):
        """Promote an overlay to LOCKED status."""
        if glyph in self.db:
            self.db[glyph]["status"] = "LOCKED"
    
    def get(self, glyph: str) -> Optional[Dict]:
        """Get an overlay."""
        return self.db.get(glyph)
    
    def add_evidence(self, glyph: str, evidence: Dict):
        """Add evidence for an overlay."""
        if glyph in self.db:
            self.db[glyph]["evidence"].append(evidence)


class HyperpermOracle:
    """Oracle for hyperpermutation ordering."""
    
    def __init__(self):
        self.items: Dict[str, Dict] = {}
    
    def _key(self, atoms: List[str]) -> str:
        """Generate key for atom set."""
        return "|".join(sorted(atoms))
    
    def add_order(
        self,
        atoms: List[str],
        seq: List[str],
        channel: str
    ) -> Dict:
        """Add an ordering for a set of atoms."""
        key = self._key(atoms)
        
        if key not in self.items:
            self.items[key] = {"orders": [], "locked": False, "sigs": set()}
        
        sig = hashlib.sha256(("::".join(seq) + "||" + channel).encode()).hexdigest()
        
        if sig not in self.items[key]["sigs"]:
            self.items[key]["orders"].append({
                "sequence": seq,
                "channel": channel,
                "sig": sig
            })
            self.items[key]["sigs"].add(sig)
            
            # Lock after 8 unique orderings
            if len(self.items[key]["sigs"]) >= 8:
                self.items[key]["locked"] = True
        
        return self.items[key]
    
    def is_locked(self, atoms: List[str]) -> bool:
        """Check if atom set is locked."""
        key = self._key(atoms)
        return self.items.get(key, {}).get("locked", False)


# Global instances
_global_registry: Optional[GlyphRegistry] = None
_global_calculus: Optional[GlyphCalculus] = None


def get_glyph_registry() -> GlyphRegistry:
    """Get the global glyph registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = GlyphRegistry()
    return _global_registry


def get_glyph_calculus() -> GlyphCalculus:
    """Get the global glyph calculus."""
    global _global_calculus
    if _global_calculus is None:
        _global_calculus = GlyphCalculus()
    return _global_calculus
