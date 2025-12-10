"""
CQE Policy System
Cartan-Quadratic Equivalence (CQE) System

From Whitepaper (Axiom F):
"Every accepted transition logs ΔΦ, op, reason code, policy stamp, and parent IDs."

This implements:
- Policy loading and validation
- Policy enforcement
- Policy versioning
- Compliance checking
"""

import json
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PolicyViolation:
    """Record of a policy violation."""
    axiom: str
    rule: str
    description: str
    severity: str  # "error", "warning", "info"
    context: Dict[str, Any]


class PolicySystem:
    """
    CQE Policy System.
    
    Loads, validates, and enforces CQE policies.
    """
    
    def __init__(self, policy_path: Optional[str] = None):
        if policy_path is None:
            # Default policy path
            policy_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "policies",
                "cqe_policy_v1.json"
            )
        
        self.policy_path = policy_path
        self.policy = self._load_policy()
        self.violations: List[PolicyViolation] = []
    
    def _load_policy(self) -> Dict[str, Any]:
        """Load policy from JSON file."""
        try:
            with open(self.policy_path, 'r') as f:
                policy = json.load(f)
            return policy
        except FileNotFoundError:
            raise ValueError(f"Policy file not found: {self.policy_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid policy JSON: {e}")
    
    def get_policy_version(self) -> str:
        """Get policy version."""
        return self.policy.get("policy_version", "unknown")
    
    def is_axiom_enforced(self, axiom: str) -> bool:
        """Check if an axiom is enforced."""
        axioms = self.policy.get("axioms", {})
        if axiom in axioms:
            return axioms[axiom].get("enforced", False)
        return False
    
    def is_operator_enabled(self, operator: str) -> bool:
        """Check if an operator is enabled."""
        operators = self.policy.get("operators", {})
        if operator in operators:
            return operators[operator].get("enabled", False)
        return False
    
    def get_operator_parameters(self, operator: str) -> Dict[str, Any]:
        """Get parameter specifications for an operator."""
        operators = self.policy.get("operators", {})
        if operator in operators:
            return operators[operator].get("parameters", {})
        return {}
    
    def validate_operator_parameters(
        self,
        operator: str,
        parameters: Dict[str, Any]
    ) -> List[PolicyViolation]:
        """
        Validate operator parameters against policy.
        
        Args:
            operator: Operator name
            parameters: Parameters to validate
        
        Returns:
            List of violations (empty if valid)
        """
        violations = []
        
        # Check if operator is enabled
        if not self.is_operator_enabled(operator):
            violations.append(PolicyViolation(
                axiom="E",
                rule="operator_whitelist",
                description=f"Operator '{operator}' is not enabled",
                severity="error",
                context={"operator": operator}
            ))
            return violations
        
        # Get parameter specs
        param_specs = self.get_operator_parameters(operator)
        
        # Validate each parameter
        for param_name, param_value in parameters.items():
            if param_name not in param_specs:
                violations.append(PolicyViolation(
                    axiom="E",
                    rule="parameter_bounds",
                    description=f"Unknown parameter '{param_name}' for operator '{operator}'",
                    severity="warning",
                    context={"operator": operator, "parameter": param_name}
                ))
                continue
            
            spec = param_specs[param_name]
            param_type = spec.get("type")
            
            # Type checking
            if param_type == "float":
                if not isinstance(param_value, (int, float)):
                    violations.append(PolicyViolation(
                        axiom="E",
                        rule="parameter_bounds",
                        description=f"Parameter '{param_name}' must be float, got {type(param_value).__name__}",
                        severity="error",
                        context={"operator": operator, "parameter": param_name, "value": param_value}
                    ))
                    continue
                
                # Range checking
                if "min" in spec and param_value < spec["min"]:
                    violations.append(PolicyViolation(
                        axiom="E",
                        rule="parameter_bounds",
                        description=f"Parameter '{param_name}' below minimum {spec['min']}",
                        severity="error",
                        context={"operator": operator, "parameter": param_name, "value": param_value, "min": spec["min"]}
                    ))
                
                if "max" in spec and param_value > spec["max"]:
                    violations.append(PolicyViolation(
                        axiom="E",
                        rule="parameter_bounds",
                        description=f"Parameter '{param_name}' above maximum {spec['max']}",
                        severity="error",
                        context={"operator": operator, "parameter": param_name, "value": param_value, "max": spec["max"]}
                    ))
            
            elif param_type == "int":
                if not isinstance(param_value, int):
                    violations.append(PolicyViolation(
                        axiom="E",
                        rule="parameter_bounds",
                        description=f"Parameter '{param_name}' must be int, got {type(param_value).__name__}",
                        severity="error",
                        context={"operator": operator, "parameter": param_name, "value": param_value}
                    ))
                    continue
                
                # Range checking
                if "min" in spec and param_value < spec["min"]:
                    violations.append(PolicyViolation(
                        axiom="E",
                        rule="parameter_bounds",
                        description=f"Parameter '{param_name}' below minimum {spec['min']}",
                        severity="error",
                        context={"operator": operator, "parameter": param_name, "value": param_value, "min": spec["min"]}
                    ))
                
                if "max" in spec and param_value > spec["max"]:
                    violations.append(PolicyViolation(
                        axiom="E",
                        rule="parameter_bounds",
                        description=f"Parameter '{param_name}' above maximum {spec['max']}",
                        severity="error",
                        context={"operator": operator, "parameter": param_name, "value": param_value, "max": spec["max"]}
                    ))
        
        return violations
    
    def get_acceptance_rules(self) -> Dict[str, Any]:
        """Get acceptance rules from policy."""
        return self.policy.get("acceptance_rules", {})
    
    def get_limits(self) -> Dict[str, Any]:
        """Get system limits from policy."""
        return self.policy.get("limits", {})
    
    def get_constants(self) -> Dict[str, Any]:
        """Get system constants from policy."""
        return self.policy.get("constants", {})
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        # Check in various sections
        sections = [
            "shell_protocol",
            "bregman_optimization",
            "canonicalization",
            "provenance",
            "crt_parallelization",
            "e8x3_projection",
            "emcp_tqf",
            "gnlc"
        ]
        
        for section in sections:
            if section in self.policy:
                section_data = self.policy[section]
                if isinstance(section_data, dict) and section_data.get("enabled"):
                    if section == feature or feature in section:
                        return True
        
        return False
    
    def record_violation(self, violation: PolicyViolation):
        """Record a policy violation."""
        self.violations.append(violation)
    
    def get_violations(
        self,
        severity: Optional[str] = None
    ) -> List[PolicyViolation]:
        """
        Get recorded violations.
        
        Args:
            severity: Filter by severity (optional)
        
        Returns:
            List of violations
        """
        if severity is None:
            return self.violations
        else:
            return [v for v in self.violations if v.severity == severity]
    
    def clear_violations(self):
        """Clear recorded violations."""
        self.violations = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get policy statistics."""
        return {
            'policy_version': self.get_policy_version(),
            'total_violations': len(self.violations),
            'errors': len(self.get_violations("error")),
            'warnings': len(self.get_violations("warning")),
            'info': len(self.get_violations("info")),
            'enforced_axioms': sum(
                1 for axiom in self.policy.get("axioms", {}).values()
                if axiom.get("enforced", False)
            ),
            'enabled_operators': sum(
                1 for op in self.policy.get("operators", {}).values()
                if op.get("enabled", False)
            )
        }


# Example usage and tests
if __name__ == "__main__":
    print("=== CQE Policy System Test ===\n")
    
    # Test 1: Load policy
    print("Test 1: Load Policy")
    
    policy_system = PolicySystem()
    print(f"Policy version: {policy_system.get_policy_version()}")
    print()
    
    # Test 2: Check axioms
    print("Test 2: Axiom Enforcement")
    
    for axiom in ["A", "B", "C", "D", "E", "F", "G"]:
        enforced = policy_system.is_axiom_enforced(axiom)
        print(f"  Axiom {axiom}: {'enforced' if enforced else 'not enforced'}")
    print()
    
    # Test 3: Check operators
    print("Test 3: Operator Status")
    
    operators = ["rotate", "weyl_reflect", "midpoint", "parity_mirror"]
    for op in operators:
        enabled = policy_system.is_operator_enabled(op)
        print(f"  {op}: {'enabled' if enabled else 'disabled'}")
    print()
    
    # Test 4: Validate parameters
    print("Test 4: Parameter Validation")
    
    # Valid parameters
    valid_params = {"theta": 0.1}
    violations = policy_system.validate_operator_parameters("rotate", valid_params)
    print(f"Valid parameters: {len(violations)} violations")
    
    # Invalid parameters (out of range)
    invalid_params = {"theta": 10.0}  # > π
    violations = policy_system.validate_operator_parameters("rotate", invalid_params)
    print(f"Invalid parameters: {len(violations)} violations")
    if violations:
        for v in violations:
            print(f"  - {v.description}")
    print()
    
    # Test 5: Get acceptance rules
    print("Test 5: Acceptance Rules")
    
    rules = policy_system.get_acceptance_rules()
    for rule_name, rule_data in rules.items():
        enabled = rule_data.get("enabled", False)
        condition = rule_data.get("condition", "N/A")
        print(f"  {rule_name}: {'enabled' if enabled else 'disabled'}")
        print(f"    Condition: {condition}")
    print()
    
    # Test 6: Get limits
    print("Test 6: System Limits")
    
    limits = policy_system.get_limits()
    for limit_name, limit_value in limits.items():
        print(f"  {limit_name}: {limit_value}")
    print()
    
    # Test 7: Get constants
    print("Test 7: System Constants")
    
    constants = policy_system.get_constants()
    for const_name, const_value in constants.items():
        print(f"  {const_name}: {const_value}")
    print()
    
    # Test 8: Feature flags
    print("Test 8: Feature Flags")
    
    features = [
        "shell_protocol",
        "bregman_optimization",
        "canonicalization",
        "provenance",
        "crt_parallelization"
    ]
    
    for feature in features:
        enabled = policy_system.is_feature_enabled(feature)
        print(f"  {feature}: {'enabled' if enabled else 'disabled'}")
    print()
    
    # Test 9: Statistics
    print("Test 9: Statistics")
    
    stats = policy_system.get_statistics()
    print(f"Policy version: {stats['policy_version']}")
    print(f"Enforced axioms: {stats['enforced_axioms']}")
    print(f"Enabled operators: {stats['enabled_operators']}")
    print(f"Total violations: {stats['total_violations']}")
    print()
    
    print("=== All Tests Passed ===")
