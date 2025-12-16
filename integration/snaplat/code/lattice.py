"""
LATTICE Module
--------------

Contains: e8_bridge, ALENAOps, EnhancedMORSRExplorer, ShellingCompressor, E8Lattice, LambdaTerm, __init__, E8Root
"""
from pathlib import Path


import sys

try:
    import Any
except ImportError:
    Any = None
try:
    import Dict
except ImportError:
    Dict = None
try:
    import List
except ImportError:
    List = None
try:
    import Optional
except ImportError:
    Optional = None
try:
    import Tuple
except ImportError:
    Tuple = None
try:
    import Union
except ImportError:
    Union = None

try:
    import numpy as np
except ImportError:
    np = None


# ============================================================================
# e8_bridge
# ============================================================================


#!/usr/bin/env python3.11
"""
Extended Lambda Calculus (Λ⊗E₈)
================================

Lambda calculus extended to capture geometric transforms in E₈ space.
Integrates with:
- Geometric Transformer (captures transform operations as lambda)
- Token Object System (lambda IR in tokens)
- AGRM/MDHG (path operations as lambda composition)

Key features:
- Geometric operations as lambda terms
- E₈ lattice navigation as lambda composition
- Dihedral operations as lambda transformations
- Automatic derivation from system operations
- Type system for geometric constraints
"""

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================================
# LAMBDA TERM TYPES
# ============================================================================

class LambdaType(Enum):
    """"""
    SCALAR = "scalar"           # Real number
    VECTOR = "vector"           # E₈ vector
    LATTICE = "lattice"         # E₈ lattice point
    TRANSFORM = "transform"     # Geometric transform
    PATH = "path"               # AGRM path
    TOKEN = "token"             # Token Object
    DIHEDRAL = "dihedral"       # Dihedral group element

@dataclass
class LambdaTerm:
    """
    A term in the extended lambda calculus.
    
    Grammar:
        t ::= x                     (variable)
            | λ x: τ. t            (abstraction)
            | t t                   (application)
            | (e8_embed t)          (E₈ embedding)
            | (e8_project t d)      (E₈ projection to dimension d)
            | (e8_navigate t w)     (Navigate E₈ via Weyl chamber w)
            | (dihedral_op N k t)   (Dihedral operation)
            | (path_compose t₁ t₂)  (AGRM path composition)
            | (conserve t)          (Apply conservation law)
    """
    term_type: str  # "var", "abs", "app", "e8_op", "dihedral_op", "path_op"
    content: Any    # Depends on term_type
    lambda_type: Optional[LambdaType] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_string(self) -> str:
        """"""
        if self.term_type == "var":
            return self.content
        
        elif self.term_type == "abs":
            var, body = self.content
            type_annotation = f": {self.lambda_type.value}" if self.lambda_type else ""
            return f"(λ {var}{type_annotation}. {body.to_string()})"
        
        elif self.term_type == "app":
            func, arg = self.content
            return f"({func.to_string()} {arg.to_string()})"
        
        elif self.term_type == "e8_op":
            op_name, args = self.content
            arg_strs = [a.to_string() if isinstance(a, LambdaTerm) else str(a) for a in args]
            return f"({op_name} {' '.join(arg_strs)})"
        
        elif self.term_type == "dihedral_op":
            N, k, reflect, arg = self.content
            return f"(D_{N}^{k}{'*' if reflect else ''} {arg.to_string()})"
        
        elif self.term_type == "path_op":
            op_name, paths = self.content
            path_strs = [p.to_string() if isinstance(p, LambdaTerm) else str(p) for p in paths]
            return f"({op_name} {' '.join(path_strs)})"
        
        else:
            return f"<{self.term_type}>"

# ============================================================================
# LAMBDA CALCULUS BUILDER
# ============================================================================

class LambdaE8Builder:
    """
    Builder for extended lambda calculus terms.
    
    Provides high-level API for constructing lambda terms from
    geometric operations.
    """
    
    def __init__(self):
        self.term_counter = 0
        self.environment: Dict[str, LambdaTerm] = {}
    
    def fresh_var(self, prefix: str = "x") -> str:
        """"""
        self.term_counter += 1
        return f"{prefix}{self.term_counter}"
    
    def var(self, name: str, lambda_type: Optional[LambdaType] = None) -> LambdaTerm:
        """"""
        return LambdaTerm("var", name, lambda_type)
    
    def abs(self, var: str, body: LambdaTerm, lambda_type: Optional[LambdaType] = None) -> LambdaTerm:
        """"""
        return LambdaTerm("abs", (var, body), lambda_type)
    
    def app(self, func: LambdaTerm, arg: LambdaTerm) -> LambdaTerm:
        """"""
        return LambdaTerm("app", (func, arg))
    
    def e8_embed(self, term: LambdaTerm) -> LambdaTerm:
        """"""
        return LambdaTerm("e8_op", ("e8_embed", [term]), LambdaType.LATTICE)
    
    def e8_project(self, term: LambdaTerm, target_dim: int) -> LambdaTerm:
        """"""
        return LambdaTerm("e8_op", ("e8_project", [term, target_dim]), LambdaType.VECTOR)
    
    def e8_navigate(self, term: LambdaTerm, weyl_chamber: int) -> LambdaTerm:
        """"""
        return LambdaTerm("e8_op", ("e8_navigate", [term, weyl_chamber]), LambdaType.LATTICE)
    
    def dihedral(self, N: int, k: int, reflect: bool, term: LambdaTerm) -> LambdaTerm:
        """"""
        return LambdaTerm("dihedral_op", (N, k, reflect, term), LambdaType.DIHEDRAL)
    
    def path_compose(self, path1: LambdaTerm, path2: LambdaTerm) -> LambdaTerm:
        """"""
        return LambdaTerm("path_op", ("path_compose", [path1, path2]), LambdaType.PATH)
    
    def conserve(self, term: LambdaTerm) -> LambdaTerm:
        """"""
        return LambdaTerm("e8_op", ("conserve", [term]), term.lambda_type)
    
    def compose(self, *terms: LambdaTerm) -> LambdaTerm:
        """"""
        if not terms:
            # Identity function
            x = self.fresh_var()
            return self.abs(x, self.var(x))
        
        if len(terms) == 1:
            return terms[0]
        
        # Build composition: (f ∘ g)(x) = f(g(x))
        result = terms[-1]
        for term in reversed(terms[:-1]):
            x = self.fresh_var()
            result = self.abs(
                x,
                self.app(term, self.app(result, self.var(x)))
            )
        
        return result

# ============================================================================
# GEOMETRIC OPERATION CAPTURE
# ============================================================================

class GeometricLambdaCapture:
    """
    Captures geometric operations and converts them to lambda calculus.
    
    Integrates with:
    - Geometric Transformer (attention, feedforward, etc.)
    - Token Object System (tokenization operations)
    - AGRM/MDHG (path operations)
    """
    
    def __init__(self):
        self.builder = LambdaE8Builder()
        self.operation_log: List[Tuple[str, LambdaTerm]] = []
    
    def capture_attention(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        num_heads: int
    ) -> LambdaTerm:
        """
        Capture attention operation as lambda term.
        
        Attention(Q, K, V) = softmax(Q·K^T / √d) · V
        
        In lambda calculus:
        λ Q. λ K. λ V. (e8_project (softmax (scale (dot Q (transpose K)))) value_dim) · V
        """
        Q = self.builder.var("Q", LambdaType.VECTOR)
        K = self.builder.var("K", LambdaType.VECTOR)
        V = self.builder.var("V", LambdaType.VECTOR)
        
        # Q · K^T
        dot_product = LambdaTerm("e8_op", ("dot", [Q, LambdaTerm("e8_op", ("transpose", [K]))]))
        
        # Scale by √d
        scaled = LambdaTerm("e8_op", ("scale", [dot_product, 1.0 / (key_dim ** 0.5)]))
        
        # Softmax
        attention_weights = LambdaTerm("e8_op", ("softmax", [scaled]))
        
        # Apply to values
        output = LambdaTerm("e8_op", ("dot", [attention_weights, V]))
        
        # Build lambda abstraction
        lambda_term = self.builder.abs("Q",
            self.builder.abs("K",
                self.builder.abs("V", output, LambdaType.VECTOR),
                LambdaType.VECTOR),
            LambdaType.VECTOR)
        
        self.operation_log.append(("attention", lambda_term))
        return lambda_term
    
    def capture_feedforward(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> LambdaTerm:
        """
        Capture feedforward network as lambda term.
        
        FFN(x) = W₂ · gelu(W₁ · x)
        
        In lambda calculus:
        λ x. (e8_project (gelu (e8_project x hidden_dim)) output_dim)
        """
        x = self.builder.var("x", LambdaType.VECTOR)
        
        # W₁ · x (project to hidden)
        hidden = self.builder.e8_project(x, hidden_dim)
        
        # gelu activation
        activated = LambdaTerm("e8_op", ("gelu", [hidden]))
        
        # W₂ · h (project to output)
        output = self.builder.e8_project(activated, output_dim)
        
        lambda_term = self.builder.abs("x", output, LambdaType.VECTOR)
        
        self.operation_log.append(("feedforward", lambda_term))
        return lambda_term
    
    def capture_layer_norm(self, dim: int) -> LambdaTerm:
        """
        Capture layer normalization as lambda term.
        
        LayerNorm(x) = (x - μ) / σ
        
        In lambda calculus:
        λ x. (e8_op normalize x)
        """
        x = self.builder.var("x", LambdaType.VECTOR)
        normalized = LambdaTerm("e8_op", ("normalize", [x]))
        
        lambda_term = self.builder.abs("x", normalized, LambdaType.VECTOR)
        
        self.operation_log.append(("layer_norm", lambda_term))
        return lambda_term
    
    def capture_tokenization(
        self,
        surface: str,
        embedding_dim: int
    ) -> LambdaTerm:
        """
        Capture tokenization as lambda term.
        
        Tokenize(text) = embed_e8(text, dim)
        
        In lambda calculus:
        λ text. (e8_embed (lookup text vocab) dim)
        """
        text = self.builder.var("text", LambdaType.SCALAR)
        
        # Lookup in vocabulary
        token_id = LambdaTerm("e8_op", ("lookup", [text, "vocab"]))
        
        # Embed in E₈
        embedded = self.builder.e8_embed(token_id)
        
        # Project to target dimension
        projected = self.builder.e8_project(embedded, embedding_dim)
        
        lambda_term = self.builder.abs("text", projected, LambdaType.TOKEN)
        
        self.operation_log.append(("tokenization", lambda_term))
        return lambda_term
    
    def capture_agrm_path(
        self,
        start_node: str,
        end_node: str,
        path_nodes: List[str]
    ) -> LambdaTerm:
        """
        Capture AGRM path as lambda term.
        
        Path(start, end) = compose(edge₁, edge₂, ..., edgeₙ)
        
        In lambda calculus:
        λ start. λ end. (path_compose edge₁ (path_compose edge₂ ... edgeₙ))
        """
        # Create edge terms
        edges = []
        for i in range(len(path_nodes) - 1):
            edge = LambdaTerm("path_op", ("edge", [path_nodes[i], path_nodes[i+1]]), LambdaType.PATH)
            edges.append(edge)
        
        # Compose edges
        if not edges:
            # Empty path (identity)
            path_term = LambdaTerm("path_op", ("identity", []), LambdaType.PATH)
        else:
            path_term = edges[0]
            for edge in edges[1:]:
                path_term = self.builder.path_compose(path_term, edge)
        
        # Build lambda abstraction
        lambda_term = self.builder.abs("start",
            self.builder.abs("end", path_term, LambdaType.PATH),
            LambdaType.PATH)
        
        self.operation_log.append(("agrm_path", lambda_term))
        return lambda_term
    
    def capture_dihedral_transform(
        self,
        N: int,
        k: int,
        reflect: bool
    ) -> LambdaTerm:
        """
        Capture dihedral group operation as lambda term.
        
        D_N^k(x) = rotate(x, 2πk/N) [with optional reflection]
        
        In lambda calculus:
        λ x. (D_N^k x)
        """
        x = self.builder.var("x", LambdaType.VECTOR)
        transformed = self.builder.dihedral(N, k, reflect, x)
        
        lambda_term = self.builder.abs("x", transformed, LambdaType.DIHEDRAL)
        
        self.operation_log.append(("dihedral", lambda_term))
        return lambda_term
    
    def get_composed_lambda(self) -> LambdaTerm:
        """"""
        if not self.operation_log:
            return self.builder.abs("x", self.builder.var("x"))
        
        terms = [term for _, term in self.operation_log]
        return self.builder.compose(*terms)
    
    def export_log(self, filepath: str):
        """"""
        log_data = [
            {
                "operation": op_name,
                "lambda_term": term.to_string(),
                "type": term.lambda_type.value if term.lambda_type else None
            }
            for op_name, term in self.operation_log
        ]
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Exported {len(log_data)} lambda operations to {filepath}")

# ============================================================================
# LAMBDA CALCULUS EVALUATOR
# ============================================================================

class LambdaE8Evaluator:
    """
    Evaluator for extended lambda calculus.
    
    Performs beta-reduction and geometric operations.
    """
    
    def __init__(self):
        self.reduction_steps = 0
        self.max_steps = 1000
    
    def evaluate(self, term: LambdaTerm, env: Dict[str, Any] = None) -> Any:
        """
        Evaluate a lambda term.
        
        Args:
            term: Lambda term to evaluate
            env: Environment mapping variables to values
            
        Returns:
            Evaluated result
        """
        if env is None:
            env = {}
        
        self.reduction_steps = 0
        return self._eval(term, env)
    
    def _eval(self, term: LambdaTerm, env: Dict[str, Any]) -> Any:
        """"""
        self.reduction_steps += 1
        
        if self.reduction_steps > self.max_steps:
            raise RuntimeError("Maximum reduction steps exceeded")
        
        if term.term_type == "var":
            return env.get(term.content, term.content)
        
        elif term.term_type == "abs":
            # Return closure
            return ("closure", term, env.copy())
        
        elif term.term_type == "app":
            func, arg = term.content
            func_val = self._eval(func, env)
            arg_val = self._eval(arg, env)
            
            if isinstance(func_val, tuple) and func_val[0] == "closure":
                _, abs_term, closure_env = func_val
                var, body = abs_term.content
                new_env = closure_env.copy()
                new_env[var] = arg_val
                return self._eval(body, new_env)
            else:
                return ("app", func_val, arg_val)
        
        elif term.term_type == "e8_op":
            op_name, args = term.content
            eval_args = [self._eval(a, env) if isinstance(a, LambdaTerm) else a for a in args]
            return (f"e8_{op_name}", *eval_args)
        
        elif term.term_type == "dihedral_op":
            N, k, reflect, arg = term.content
            eval_arg = self._eval(arg, env)
            return ("dihedral", N, k, reflect, eval_arg)
        
        elif term.term_type == "path_op":
            op_name, paths = term.content
            eval_paths = [self._eval(p, env) if isinstance(p, LambdaTerm) else p for p in paths]
            return (f"path_{op_name}", *eval_paths)
        
        else:
            return term

# ============================================================================
# DEMO
# ============================================================================

def demo_lambda_e8_calculus():
    """"""
    print("="*70)
    print("EXTENDED LAMBDA CALCULUS (Λ⊗E₈) DEMO")
    print("="*70)
    
    capture = GeometricLambdaCapture()
    
    # Capture various operations
    print("\\n[1] Capturing geometric operations...")
    
    attention = capture.capture_attention(1024, 1024, 1024, 16)
    print(f"\\nAttention: {attention.to_string()}")
    
    ffn = capture.capture_feedforward(1024, 4096, 1024)
    print(f"\\nFeedforward: {ffn.to_string()}")
    
    norm = capture.capture_layer_norm(1024)
    print(f"\\nLayer Norm: {norm.to_string()}")
    
    tokenize = capture.capture_tokenization("hello", 320000)
    print(f"\\nTokenization: {tokenize.to_string()}")
    
    path = capture.capture_agrm_path("A", "D", ["A", "B", "C", "D"])
    print(f"\\nAGRM Path: {path.to_string()}")
    
    dihedral = capture.capture_dihedral_transform(12, 3, False)
    print(f"\\nDihedral: {dihedral.to_string()}")
    
    # Compose all operations
    print("\\n" + "="*70)
    print("[2] Composing all operations...")
    
    composed = capture.get_composed_lambda()
    print(f"\\nComposed lambda: {composed.to_string()}")
    
    # Export log
    capture.export_log(str(Path(__file__).parent / "lambda_operations_log.json"))
    
    # Demonstrate evaluation
    print("\\n" + "="*70)
    print("[3] Evaluating lambda terms...")
    
    evaluator = LambdaE8Evaluator()
    
    # Simple example: (λ x. x) 42
    builder = LambdaE8Builder()
    identity = builder.abs("x", builder.var("x"))
    result = evaluator.evaluate(builder.app(identity, builder.var("42")))
    print(f"\\n(λ x. x) 42 = {result}")
    print(f"Reduction steps: {evaluator.reduction_steps}")
    
    print("\\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)

if __name__ == "__main__":
    demo_lambda_e8_calculus()




# ============================================================================
# ALENAOps
# ============================================================================

class ALENAOps:
    """ALENA tensor operations for curvature projection."""
    
    def __init__(self, e8_lattice: E8Lattice):
        self.e8 = e8_lattice
        
    def r_theta_snap(self, vector: np.ndarray) -> np.ndarray:
        """Snap to nearest Rθ position (polar snap)."""
        # Convert to polar coordinates
        r = np.linalg.norm(vector)
        
        # Snap radius to Fibonacci lattice
        fib_radii = [PHI**n * COUPLING for n in range(-10, 10)]
        nearest_r = min(fib_radii, key=lambda x: abs(x - r))
        
        # Normalize and scale
        if r > 0:
            snapped = vector / r * nearest_r
        else:
            snapped = vector
        
        return snapped
    
    def weyl_flip(self, vector: np.ndarray) -> np.ndarray:
        """Flip across Weyl chamber boundary."""
        chamber = self.e8.find_weyl_chamber(vector)
        normal = self.e8.weyl_chambers[chamber]
        
        # Reflect across hyperplane
        flipped = vector - 2 * np.dot(vector, normal) * normal
        
        return self.e8.project_to_manifold(flipped)
    
    def midpoint_ecc(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Midpoint with error-correcting code."""
        # Compute midpoint
        mid = (v1 + v2) / 2
        
        # Project to E8 lattice for error correction
        corrected = self.e8.project_to_lattice(mid)
        
        return corrected
    
    def project_curvature(self, vector: np.ndarray, 
                         face_angle: float = 0.0) -> np.ndarray:
        """
        Project E8 face to show curvature on flat surface.
        This is the key ALENA operation - creates spacetime curvature.
        """
        # Rotate face
        rotated = self.e8.face_rotation(vector, face_angle)
        
        # Project to lower dimensions (creates curvature effect)
        # Use stereographic projection from E8 to R^7
        if abs(rotated[7] - 1.0) < 1e-6:
            # Avoid singularity at north pole
            projected = rotated[:7]
        else:
            scale = 1.0 / (1.0 - rotated[7])
            projected = rotated[:7] * scale
        
        # Embed back into E8 with curvature information
        curved = np.zeros(8)
        curved[:7] = projected
        curved[7] = np.linalg.norm(projected) * COUPLING  # Curvature measure
        
        return self.e8.project_to_manifold(curved)


# ============================================================================
# EnhancedMORSRExplorer
# ============================================================================

class EnhancedMORSRExplorer:
    """Enhanced MORSR Explorer with dynamic pulse adjustments for lattice optimization."""
    def __init__(self):
        self.radius = MORSR_RADIUS
        self.dwell = MORSR_DWELL
        self.best_score = 0.0

    @ladder_hook
    def explore(self, vector: np.ndarray) -> Tuple[np.ndarray, float]:
        """Explore lattice with MORSR pulses, adjust radius for best score."""
        best_vector = vector.copy()
        for radius in range(5, 10):
            pulsed = vector.copy()
            for _ in range(self.dwell):
                for i in range(len(pulsed)):
                    if i % 2 == 0:
                        pulsed[i] *= radius
                    else:
                        pulsed[i] = -pulsed[i]
            score = sp_norm(pulsed) / sp_norm(vector) if sp_norm(vector) > 0 else 1.0
            if score > self.best_score:
                self.best_score = score
                best_vector = pulsed
        return best_vector, self.best_score

    def morsr_pulse(self, vector: np.ndarray) -> np.ndarray:
        """Apply MORSR pulses for ΔΦ≤0 snap with dynamic adjustment."""
        for _ in range(self.dwell):
            for i in range(len(vector)):
                if i % 2 == 0:
                    vector[i] = vector[i] * self.radius
                else:
                    vector[i] = -vector[i]
        return vector




# ============================================================================
# ShellingCompressor
# ============================================================================

class ShellingCompressor:
    """Shelling Compressor: n=1-10 triad/inverse glyphs with Cartan path integration."""
    def __init__(self, levels=10):
        self.levels = levels
        self.glyphs = {}

    @ladder_hook
    def compress_to_glyph(self, text: str, level: int = 1) -> str:
        """Compress text into triad/inverse glyphs for Cartan path representation."""
        words = text.lower().split()
        triad = ' '.join(words[:3]) if len(words) >= 3 else ' '.join(words)
        inverse = ' '.join(words[-3:][::-1]) if len(words) >= 3 else triad[::-1]
        glyph = f"{triad}|{inverse}"
        self.glyphs[text[:10]] = glyph
        return glyph if level <= self.levels else text




# ============================================================================
# E8Lattice
# ============================================================================

class E8Lattice:
    """E8 exceptional Lie group lattice operations."""
    
    def __init__(self):
        self.roots = self._generate_roots()
        self.weyl_chambers = self._generate_weyl_chambers()
        
    def _generate_roots(self) -> List[E8Root]:
        """Generate all 240 E8 root vectors."""
        roots = []
        
        # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations (112 roots)
        for i in range(8):
            for j in range(i+1, 8):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        coords = np.zeros(8)
                        coords[i] = s1
                        coords[j] = s2
                        roots.append(E8Root(coords, len(roots), E8_NORM))
        
        # Type 2: (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) 
        # with even number of minus signs (128 roots)
        for signs in range(256):
            coords = np.array([(1 if (signs >> i) & 1 else -1) / 2 
                              for i in range(8)])
            if np.sum(coords < 0) % 2 == 0:  # Even number of minus signs
                roots.append(E8Root(coords, len(roots), E8_NORM))
        
        return roots[:240]  # Ensure exactly 240 roots
    
    def _generate_weyl_chambers(self) -> List[np.ndarray]:
        """Generate 48 Weyl chambers (fundamental domains)."""
        chambers = []
        
        # Weyl group of E8 has order 696,729,600
        # We use 48 fundamental chambers for practical purposes
        for i in range(48):
            # Each chamber is a cone in E8 space
            # Defined by hyperplane normals
            angle = (2 * np.pi * i) / 48
            normal = np.array([
                np.cos(angle),
                np.sin(angle),
                np.cos(2*angle),
                np.sin(2*angle),
                np.cos(3*angle),
                np.sin(3*angle),
                np.cos(4*angle),
                np.sin(4*angle)
            ])
            chambers.append(normal / np.linalg.norm(normal))
        
        return chambers
    
    def project_to_lattice(self, vector: np.ndarray) -> np.ndarray:
        """Project vector to nearest E8 lattice point."""
        # Find nearest root
        distances = [np.linalg.norm(vector - root.coords) 
                    for root in self.roots]
        nearest_idx = np.argmin(distances)
        return self.roots[nearest_idx].coords
    
    def project_to_manifold(self, vector: np.ndarray) -> np.ndarray:
        """Project to continuous E8 manifold (unit sphere)."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm * E8_NORM
        return vector
    
    def find_weyl_chamber(self, vector: np.ndarray) -> int:
        """Find which Weyl chamber contains the vector."""
        # Compute dot product with each chamber normal
        dots = [np.dot(vector, chamber) for chamber in self.weyl_chambers]
        return np.argmax(dots)
    
    def interpolate_geodesic(self, start: np.ndarray, end: np.ndarray, 
                            t: float) -> np.ndarray:
        """Interpolate along geodesic on E8 manifold."""
        # Spherical linear interpolation (SLERP)
        dot = np.dot(start, end) / (np.linalg.norm(start) * np.linalg.norm(end))
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)
        
        if abs(theta) < 1e-6:
            # Vectors are parallel, use linear interpolation
            return (1 - t) * start + t * end
        
        sin_theta = np.sin(theta)
        a = np.sin((1 - t) * theta) / sin_theta
        b = np.sin(t * theta) / sin_theta
        
        result = a * start + b * end
        return self.project_to_manifold(result)
    
    def rotate_e8(self, vector: np.ndarray, axis1: int, axis2: int, 
                  angle: float) -> np.ndarray:
        """Rotate vector in E8 space around plane defined by axis1, axis2."""
        result = vector.copy()
        
        # 2D rotation in the specified plane
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        x = result[axis1]
        y = result[axis2]
        
        result[axis1] = cos_a * x - sin_a * y
        result[axis2] = sin_a * x + cos_a * y
        
        return result
    
    def face_rotation(self, vector: np.ndarray, angle: float) -> np.ndarray:
        """Rotate E8 face - generates different solution paths (P vs NP)."""
        # Rotate in multiple planes simultaneously
        result = vector.copy()
        
        # Primary rotation (0-1 plane)
        result = self.rotate_e8(result, 0, 1, angle)
        
        # Secondary rotation (2-3 plane)
        result = self.rotate_e8(result, 2, 3, angle * PHI)
        
        # Tertiary rotation (4-5 plane)
        result = self.rotate_e8(result, 4, 5, angle * PHI**2)
        
        # Quaternary rotation (6-7 plane)
        result = self.rotate_e8(result, 6, 7, angle * PHI**3)
        
        return self.project_to_manifold(result)
    
    def compute_digital_root(self, vector: np.ndarray) -> int:
        """Compute digital root (0-9) from E8 vector."""
        # Sum all components, reduce to single digit
        total = int(np.sum(np.abs(vector)) * 1000)  # Scale for precision
        while total >= 10:
            total = sum(int(d) for d in str(total))
        return total if total > 0 else 9
    
    def compute_parity_channels(self, vector: np.ndarray) -> np.ndarray:
        """Compute 24 parity channels from E8 vector."""
        # Use Leech lattice embedding (24D)
        channels = np.zeros(24)
        
        # Embed E8 into first 8 channels
        channels[:8] = vector
        
        # Generate remaining 16 channels via modular arithmetic
        for i in range(8, 24):
            # Use CRT rails (3, 6, 9) and coupling (0.03)
            mod = (i % 3) + 3  # Moduli: 3, 4, 5, 3, 4, 5, ...
            channels[i] = (np.sum(vector) * COUPLING * i) % mod
        
        return channels


# ============================================================================
# LambdaTerm
# ============================================================================

class LambdaTerm:
    """CQE proto-language lambda calculus term represented as glyph + vector embeddings."""
    def __init__(self, expr: str, shelling: ShellingCompressor, alena: ALENAOps, morsr: EnhancedMORSRExplorer):
        self.expr = expr
        self.shelling = shelling
        self.alena = alena
        self.morsr = morsr
        self.glyph_seq = self.shelling.compress_to_glyph(expr, level=3)
        self.vector = self.text_to_vector(self.glyph_seq)

    def text_to_vector(self, text: str) -> np.ndarray:
        embed_dim = 128
        words = text.split()
        vec = np.bincount([hash(w) % embed_dim for w in words], minlength=embed_dim) / max(len(words), 1)
        norm_vec = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
        return norm_vec

    def apply(self, arg: 'LambdaTerm') -> 'LambdaTerm':
        """Apply lambda term to argument."""
        combined_expr = f"({self.expr}) ({arg.expr})"
        combined_glyph = f"{self.glyph_seq}|{arg.glyph_seq}"
        combined_vector = self.vector + arg.vector
        combined_vector = combined_vector / np.linalg.norm(combined_vector) if np.linalg.norm(combined_vector) > 0 else combined_vector
        snapped = self.alena.r_theta_snap(combined_vector)
        pulsed, _ = self.morsr.explore(np.copy(snapped))
        new_term = LambdaTerm(combined_expr, self.shelling, self.alena, self.morsr)
        new_term.glyph_seq = combined_glyph
        new_term.vector = pulsed
        return new_term

    def reduce(self) -> 'LambdaTerm':
        """Simulate reduction step."""
        flipped = self.alena.weyl_flip(self.vector)
        mid = (self.vector + flipped) / 2
        norm_mid = mid * (E8_NORM / np.linalg.norm(mid)) if np.linalg.norm(mid) > 0 else mid
        reduced_term = LambdaTerm(self.expr, self.shelling, self.alena, self.morsr)
        reduced_term.glyph_seq = self.glyph_seq
        reduced_term.vector = norm_mid
        return reduced_term




# ============================================================================
# __init__
# ============================================================================


__all__=["ast","typesys","eval","typing","modal","glyphs","e8_bridge","runtime"]
__version__="1.0.0"




# ============================================================================
# E8Root
# ============================================================================

class E8Root:
    """E8 root vector."""
    coords: np.ndarray  # (8,) coordinates
    index: int  # Root index [0-239]
    norm: float  # Norm (should be √2)
    
    def __post_init__(self):
        if self.norm is None:
            self.norm = np.linalg.norm(self.coords)

