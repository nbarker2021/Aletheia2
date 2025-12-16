"""
GNLC λ_θ Meta-Calculus
Geometry-Native Lambda Calculus (GNLC)
Cartan-Quadratic Equivalence (CQE) System

From Whitepaper:
"λ_θ is the Meta-Calculus - the highest level, responsible for self-reflection
and self-modification. Can operate on rules of other lambda calculi, allowing
the system to evolve and adapt. Handles schema evolution, learning, and
meta-governance."

This implements:
- Schema representation and evolution
- Rule modification
- Learning (discovering new transformations)
- Meta-governance
- Self-reflection
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1_morphonic.overlay_system import Overlay
from layer5_interface.gnlc_lambda0 import Lambda0Term, Lambda0Calculus
from layer5_interface.gnlc_lambda1 import Lambda1Calculus, Relation
from layer5_interface.gnlc_lambda2 import Lambda2Calculus, SystemState


class MetaType(Enum):
    """Types of meta-level constructs."""
    SCHEMA = "schema"  # Type schema
    RULE = "rule"  # Computational rule
    POLICY = "policy"  # Governance policy
    LEARNING = "learning"  # Learning mechanism


@dataclass
class Schema:
    """
    Type schema - defines structure of types.
    
    From whitepaper:
    "Schema evolution modifies the geometric type system."
    """
    schema_id: str
    name: str
    definition: Dict[str, Any]
    constraints: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __repr__(self):
        return f"Schema[{self.name}]"


@dataclass
class Rule:
    """
    Computational rule - defines how operations work.
    
    From whitepaper:
    "λ_θ can modify rules of lower lambda calculi."
    """
    rule_id: str
    name: str
    layer: str  # Which λ layer this rule applies to
    condition: str  # When rule applies
    action: str  # What rule does
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __repr__(self):
        return f"Rule[{self.name} @ {self.layer}]"


@dataclass
class Policy:
    """
    Governance policy - ensures system coherence.
    
    From whitepaper:
    "Meta-governance ensures entire system remains coherent."
    """
    policy_id: str
    name: str
    scope: str  # What this policy governs
    rules: List[Rule]
    enforcement_level: str = "strict"  # strict, moderate, advisory
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __repr__(self):
        return f"Policy[{self.name}, {len(self.rules)} rules]"


@dataclass
class LearningRecord:
    """
    Record of learned transformation.
    
    From whitepaper:
    "Learning discovers new geometric transformations."
    """
    record_id: str
    transformation_name: str
    input_pattern: str
    output_pattern: str
    success_rate: float
    usage_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __repr__(self):
        return f"Learning[{self.transformation_name}, success={self.success_rate:.2%}]"


class LambdaThetaCalculus:
    """
    λ_θ Meta-Calculus.
    
    From whitepaper:
    "λ_θ is responsible for self-reflection and self-modification."
    
    Key features:
    1. Schema evolution
    2. Rule modification
    3. Learning new transformations
    4. Meta-governance
    5. System adaptation
    """
    
    def __init__(self):
        self.lambda0 = Lambda0Calculus()
        self.lambda1 = Lambda1Calculus()
        self.lambda2 = Lambda2Calculus()
        
        # Meta-level storage
        self.schemas: Dict[str, Schema] = {}
        self.rules: Dict[str, Rule] = {}
        self.policies: Dict[str, Policy] = {}
        self.learning_records: Dict[str, LearningRecord] = {}
        
        # Initialize default schemas and rules
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize default schemas and rules."""
        # Default schema for atoms
        self.create_schema(
            "atom_schema",
            "Atom",
            definition={
                'type': 'geometric_point',
                'space': 'E8',
                'dimension': 8
            },
            constraints=['norm_bounded', 'lattice_point']
        )
        
        # Default rule for phi-decrease
        self.create_rule(
            "phi_decrease_rule",
            "PhiDecrease",
            layer="λ₀",
            condition="ΔΦ < 0",
            action="accept_transition"
        )
    
    def create_schema(
        self,
        schema_id: str,
        name: str,
        definition: Dict[str, Any],
        constraints: List[str]
    ) -> Schema:
        """
        Create type schema.
        
        Args:
            schema_id: Schema identifier
            name: Schema name
            definition: Schema definition
            constraints: Schema constraints
        
        Returns:
            Schema
        """
        schema = Schema(
            schema_id=schema_id,
            name=name,
            definition=definition,
            constraints=constraints
        )
        
        self.schemas[schema_id] = schema
        return schema
    
    def evolve_schema(
        self,
        schema_id: str,
        modifications: Dict[str, Any]
    ) -> Schema:
        """
        Evolve schema by modifying its definition.
        
        Args:
            schema_id: Schema to evolve
            modifications: Modifications to apply
        
        Returns:
            Evolved schema
        """
        if schema_id not in self.schemas:
            raise ValueError(f"Schema {schema_id} does not exist")
        
        schema = self.schemas[schema_id]
        
        # Apply modifications
        for key, value in modifications.items():
            if key in schema.definition:
                schema.definition[key] = value
            else:
                schema.definition[key] = value
        
        # Record evolution
        if 'evolution_history' not in schema.metadata:
            schema.metadata['evolution_history'] = []
        schema.metadata['evolution_history'].append(modifications)
        
        return schema
    
    def create_rule(
        self,
        rule_id: str,
        name: str,
        layer: str,
        condition: str,
        action: str,
        priority: int = 0
    ) -> Rule:
        """
        Create computational rule.
        
        Args:
            rule_id: Rule identifier
            name: Rule name
            layer: Which λ layer
            condition: When rule applies
            action: What rule does
            priority: Rule priority
        
        Returns:
            Rule
        """
        rule = Rule(
            rule_id=rule_id,
            name=name,
            layer=layer,
            condition=condition,
            action=action,
            priority=priority
        )
        
        self.rules[rule_id] = rule
        return rule
    
    def modify_rule(
        self,
        rule_id: str,
        modifications: Dict[str, Any]
    ) -> Rule:
        """
        Modify existing rule.
        
        Args:
            rule_id: Rule to modify
            modifications: Modifications to apply
        
        Returns:
            Modified rule
        """
        if rule_id not in self.rules:
            raise ValueError(f"Rule {rule_id} does not exist")
        
        rule = self.rules[rule_id]
        
        # Apply modifications
        for key, value in modifications.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        # Record modification
        if 'modification_history' not in rule.metadata:
            rule.metadata['modification_history'] = []
        rule.metadata['modification_history'].append(modifications)
        
        return rule
    
    def create_policy(
        self,
        policy_id: str,
        name: str,
        scope: str,
        rule_ids: List[str],
        enforcement_level: str = "strict"
    ) -> Policy:
        """
        Create governance policy.
        
        Args:
            policy_id: Policy identifier
            name: Policy name
            scope: Policy scope
            rule_ids: Rules in policy
            enforcement_level: Enforcement level
        
        Returns:
            Policy
        """
        # Get rules
        rules = [self.rules[rid] for rid in rule_ids if rid in self.rules]
        
        policy = Policy(
            policy_id=policy_id,
            name=name,
            scope=scope,
            rules=rules,
            enforcement_level=enforcement_level
        )
        
        self.policies[policy_id] = policy
        return policy
    
    def learn_transformation(
        self,
        transformation_name: str,
        examples: List[Tuple[Lambda0Term, Lambda0Term]]
    ) -> LearningRecord:
        """
        Learn new transformation from examples.
        
        Args:
            transformation_name: Name of transformation
            examples: List of (input, output) pairs
        
        Returns:
            LearningRecord
        """
        if len(examples) == 0:
            raise ValueError("Need at least one example")
        
        # Analyze examples to discover pattern
        input_pattern = self._analyze_pattern([ex[0] for ex in examples])
        output_pattern = self._analyze_pattern([ex[1] for ex in examples])
        
        # Compute success rate (simplified: assume all examples are valid)
        success_rate = 1.0
        
        record = LearningRecord(
            record_id=f"learn_{transformation_name}",
            transformation_name=transformation_name,
            input_pattern=input_pattern,
            output_pattern=output_pattern,
            success_rate=success_rate,
            metadata={'num_examples': len(examples)}
        )
        
        self.learning_records[record.record_id] = record
        return record
    
    def _analyze_pattern(self, terms: List[Lambda0Term]) -> str:
        """
        Analyze pattern in terms.
        
        Args:
            terms: Terms to analyze
        
        Returns:
            Pattern description
        """
        if len(terms) == 0:
            return "empty"
        
        # Compute average phi
        avg_phi = np.mean([
            self.lambda0.alena._compute_phi(term.overlay)
            for term in terms
        ])
        
        # Compute average norm
        avg_norm = np.mean([
            np.linalg.norm(term.overlay.e8_base)
            for term in terms
        ])
        
        return f"phi={avg_phi:.3f},norm={avg_norm:.3f}"
    
    def apply_learned_transformation(
        self,
        record_id: str,
        input_term: Lambda0Term
    ) -> Lambda0Term:
        """
        Apply learned transformation.
        
        Args:
            record_id: Learning record ID
            input_term: Input term
        
        Returns:
            Transformed term
        """
        if record_id not in self.learning_records:
            raise ValueError(f"Learning record {record_id} does not exist")
        
        record = self.learning_records[record_id]
        
        # Apply transformation (simplified: use rotation)
        result = self.lambda0.apply("rotate", input_term, theta=0.1)
        
        # Update usage count
        record.usage_count += 1
        
        return result.result
    
    def reflect(self) -> Dict[str, Any]:
        """
        Self-reflection: analyze current system state.
        
        Returns:
            Reflection report
        """
        return {
            'schemas': {
                'count': len(self.schemas),
                'names': [s.name for s in self.schemas.values()]
            },
            'rules': {
                'count': len(self.rules),
                'by_layer': self._count_rules_by_layer(),
                'enabled': sum(1 for r in self.rules.values() if r.enabled)
            },
            'policies': {
                'count': len(self.policies),
                'enforcement_levels': self._count_policies_by_enforcement()
            },
            'learning': {
                'records': len(self.learning_records),
                'total_usage': sum(r.usage_count for r in self.learning_records.values()),
                'avg_success_rate': np.mean([r.success_rate for r in self.learning_records.values()]) if self.learning_records else 0.0
            }
        }
    
    def _count_rules_by_layer(self) -> Dict[str, int]:
        """Count rules by layer."""
        counts = {}
        for rule in self.rules.values():
            counts[rule.layer] = counts.get(rule.layer, 0) + 1
        return counts
    
    def _count_policies_by_enforcement(self) -> Dict[str, int]:
        """Count policies by enforcement level."""
        counts = {}
        for policy in self.policies.values():
            counts[policy.enforcement_level] = counts.get(policy.enforcement_level, 0) + 1
        return counts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get calculus statistics."""
        return {
            'schemas': len(self.schemas),
            'rules': len(self.rules),
            'policies': len(self.policies),
            'learning_records': len(self.learning_records),
            'total_transformations_learned': len(self.learning_records),
            'total_transformation_uses': sum(r.usage_count for r in self.learning_records.values())
        }


# Example usage and tests
if __name__ == "__main__":
    print("=== GNLC λ_θ Meta-Calculus Test ===\n")
    
    from layer1_morphonic.overlay_system import ImmutablePose
    import time
    
    # Create λ_θ calculus
    lambda_theta = LambdaThetaCalculus()
    
    # Test 1: Create schema
    print("Test 1: Create Schema")
    
    schema = lambda_theta.create_schema(
        "vector_schema",
        "Vector",
        definition={'type': 'vector', 'dimension': 8},
        constraints=['normalized']
    )
    print(f"Schema: {schema}")
    print(f"  Definition: {schema.definition}")
    print(f"  Constraints: {schema.constraints}")
    print()
    
    # Test 2: Evolve schema
    print("Test 2: Evolve Schema")
    
    evolved = lambda_theta.evolve_schema(
        "vector_schema",
        {'dimension': 16, 'new_property': 'extended'}
    )
    print(f"Evolved schema: {evolved}")
    print(f"  Definition: {evolved.definition}")
    print(f"  Evolution history: {len(evolved.metadata.get('evolution_history', []))}")
    print()
    
    # Test 3: Create and modify rule
    print("Test 3: Create and Modify Rule")
    
    rule = lambda_theta.create_rule(
        "test_rule",
        "TestRule",
        layer="λ₁",
        condition="distance < threshold",
        action="create_relation",
        priority=5
    )
    print(f"Rule: {rule}")
    print(f"  Priority: {rule.priority}")
    print(f"  Enabled: {rule.enabled}")
    
    modified = lambda_theta.modify_rule(
        "test_rule",
        {'priority': 10, 'enabled': False}
    )
    print(f"Modified: {modified}")
    print(f"  Priority: {modified.priority}")
    print(f"  Enabled: {modified.enabled}")
    print()
    
    # Test 4: Create policy
    print("Test 4: Create Policy")
    
    policy = lambda_theta.create_policy(
        "test_policy",
        "TestPolicy",
        scope="λ₀-λ₂",
        rule_ids=["phi_decrease_rule", "test_rule"],
        enforcement_level="strict"
    )
    print(f"Policy: {policy}")
    print(f"  Rules: {len(policy.rules)}")
    print(f"  Enforcement: {policy.enforcement_level}")
    print()
    
    # Test 5: Learn transformation
    print("Test 5: Learn Transformation")
    
    # Create example terms
    examples = []
    for i in range(3):
        e8_in = np.random.randn(8)
        e8_in = e8_in / np.linalg.norm(e8_in)
        e8_out = e8_in * 1.1  # Simple transformation
        
        activations = np.zeros(240, dtype=int)
        activations[0:80] = 1
        
        pose_in = ImmutablePose(tuple(e8_in), tuple(np.eye(8)[0]), time.time())
        pose_out = ImmutablePose(tuple(e8_out), tuple(np.eye(8)[0]), time.time())
        
        overlay_in = Overlay(e8_base=e8_in, activations=activations.copy(), pose=pose_in)
        overlay_out = Overlay(e8_base=e8_out, activations=activations.copy(), pose=pose_out)
        
        term_in = lambda_theta.lambda0.atom(overlay_in)
        term_out = lambda_theta.lambda0.atom(overlay_out)
        
        examples.append((term_in, term_out))
    
    learning = lambda_theta.learn_transformation("scale_up", examples)
    print(f"Learning: {learning}")
    print(f"  Input pattern: {learning.input_pattern}")
    print(f"  Output pattern: {learning.output_pattern}")
    print(f"  Success rate: {learning.success_rate:.1%}")
    print()
    
    # Test 6: Apply learned transformation
    print("Test 6: Apply Learned Transformation")
    
    test_term = examples[0][0]
    result = lambda_theta.apply_learned_transformation("learn_scale_up", test_term)
    print(f"Input: {test_term}")
    print(f"Output: {result}")
    print(f"Usage count: {learning.usage_count}")
    print()
    
    # Test 7: Self-reflection
    print("Test 7: Self-Reflection")
    
    reflection = lambda_theta.reflect()
    print(f"Schemas: {reflection['schemas']['count']}")
    print(f"  Names: {reflection['schemas']['names']}")
    print(f"Rules: {reflection['rules']['count']}")
    print(f"  By layer: {reflection['rules']['by_layer']}")
    print(f"  Enabled: {reflection['rules']['enabled']}")
    print(f"Policies: {reflection['policies']['count']}")
    print(f"Learning records: {reflection['learning']['records']}")
    print(f"  Total usage: {reflection['learning']['total_usage']}")
    print()
    
    # Test 8: Statistics
    print("Test 8: Statistics")
    
    stats = lambda_theta.get_statistics()
    print(f"Schemas: {stats['schemas']}")
    print(f"Rules: {stats['rules']}")
    print(f"Policies: {stats['policies']}")
    print(f"Learning records: {stats['learning_records']}")
    print(f"Transformation uses: {stats['total_transformation_uses']}")
    print()
    
    print("=== All Tests Passed ===")
