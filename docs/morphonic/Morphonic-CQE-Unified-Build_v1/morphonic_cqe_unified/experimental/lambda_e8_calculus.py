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

import sys
sys.path.insert(0, '/home/ubuntu/aletheia_complete_v1/core_system')

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

# ============================================================================
# LAMBDA TERM TYPES
# ============================================================================

class LambdaType(Enum):
    """Types in the extended lambda calculus."""
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
        """Convert lambda term to string representation."""
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
        """Generate a fresh variable name."""
        self.term_counter += 1
        return f"{prefix}{self.term_counter}"
    
    def var(self, name: str, lambda_type: Optional[LambdaType] = None) -> LambdaTerm:
        """Create a variable term."""
        return LambdaTerm("var", name, lambda_type)
    
    def abs(self, var: str, body: LambdaTerm, lambda_type: Optional[LambdaType] = None) -> LambdaTerm:
        """Create an abstraction (λ x. body)."""
        return LambdaTerm("abs", (var, body), lambda_type)
    
    def app(self, func: LambdaTerm, arg: LambdaTerm) -> LambdaTerm:
        """Create an application (func arg)."""
        return LambdaTerm("app", (func, arg))
    
    def e8_embed(self, term: LambdaTerm) -> LambdaTerm:
        """Embed term into E₈ lattice."""
        return LambdaTerm("e8_op", ("e8_embed", [term]), LambdaType.LATTICE)
    
    def e8_project(self, term: LambdaTerm, target_dim: int) -> LambdaTerm:
        """Project E₈ term to target dimension."""
        return LambdaTerm("e8_op", ("e8_project", [term, target_dim]), LambdaType.VECTOR)
    
    def e8_navigate(self, term: LambdaTerm, weyl_chamber: int) -> LambdaTerm:
        """Navigate E₈ lattice via Weyl chamber."""
        return LambdaTerm("e8_op", ("e8_navigate", [term, weyl_chamber]), LambdaType.LATTICE)
    
    def dihedral(self, N: int, k: int, reflect: bool, term: LambdaTerm) -> LambdaTerm:
        """Apply dihedral group operation."""
        return LambdaTerm("dihedral_op", (N, k, reflect, term), LambdaType.DIHEDRAL)
    
    def path_compose(self, path1: LambdaTerm, path2: LambdaTerm) -> LambdaTerm:
        """Compose two AGRM paths."""
        return LambdaTerm("path_op", ("path_compose", [path1, path2]), LambdaType.PATH)
    
    def conserve(self, term: LambdaTerm) -> LambdaTerm:
        """Apply conservation law (ΔΦ ≤ 0)."""
        return LambdaTerm("e8_op", ("conserve", [term]), term.lambda_type)
    
    def compose(self, *terms: LambdaTerm) -> LambdaTerm:
        """Compose multiple lambda terms (right-to-left)."""
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
        """Get the composition of all captured operations."""
        if not self.operation_log:
            return self.builder.abs("x", self.builder.var("x"))
        
        terms = [term for _, term in self.operation_log]
        return self.builder.compose(*terms)
    
    def export_log(self, filepath: str):
        """Export operation log to JSON."""
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
        """Internal evaluation with step counting."""
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
    """Demonstrate the extended lambda calculus."""
    print("="*70)
    print("EXTENDED LAMBDA CALCULUS (Λ⊗E₈) DEMO")
    print("="*70)
    
    capture = GeometricLambdaCapture()
    
    # Capture various operations
    print("\n[1] Capturing geometric operations...")
    
    attention = capture.capture_attention(1024, 1024, 1024, 16)
    print(f"\nAttention: {attention.to_string()}")
    
    ffn = capture.capture_feedforward(1024, 4096, 1024)
    print(f"\nFeedforward: {ffn.to_string()}")
    
    norm = capture.capture_layer_norm(1024)
    print(f"\nLayer Norm: {norm.to_string()}")
    
    tokenize = capture.capture_tokenization("hello", 320000)
    print(f"\nTokenization: {tokenize.to_string()}")
    
    path = capture.capture_agrm_path("A", "D", ["A", "B", "C", "D"])
    print(f"\nAGRM Path: {path.to_string()}")
    
    dihedral = capture.capture_dihedral_transform(12, 3, False)
    print(f"\nDihedral: {dihedral.to_string()}")
    
    # Compose all operations
    print("\n" + "="*70)
    print("[2] Composing all operations...")
    
    composed = capture.get_composed_lambda()
    print(f"\nComposed lambda: {composed.to_string()}")
    
    # Export log
    capture.export_log("/home/ubuntu/lambda_operations_log.json")
    
    # Demonstrate evaluation
    print("\n" + "="*70)
    print("[3] Evaluating lambda terms...")
    
    evaluator = LambdaE8Evaluator()
    
    # Simple example: (λ x. x) 42
    builder = LambdaE8Builder()
    identity = builder.abs("x", builder.var("x"))
    result = evaluator.evaluate(builder.app(identity, builder.var("42")))
    print(f"\n(λ x. x) 42 = {result}")
    print(f"Reduction steps: {evaluator.reduction_steps}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    demo_lambda_e8_calculus()

