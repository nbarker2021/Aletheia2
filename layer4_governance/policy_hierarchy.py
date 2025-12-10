"""
Policy Hierarchy - Governance Policy Management

Implements a hierarchical policy system for CQE governance.
Policies are organized by digital root (DR 0-9) and enforce
different levels of constraints.

Inspired by cqe-complete/cqe/os/governance.py
Author: Manus AI (based on CQE research)
Date: December 5, 2025
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum


class GovernanceLevel(Enum):
    """Levels of governance enforcement."""
    PERMISSIVE = "permissive"      # Minimal constraints (DR 1-3)
    STANDARD = "standard"          # Normal CQE constraints (DR 4-6)
    STRICT = "strict"              # Enhanced validation (DR 7-8)
    ULTIMATE = "ultimate"          # All constraints active (DR 9, 0)


class ConstraintType(Enum):
    """Types of constraints in CQE governance."""
    GEOMETRIC = "geometric"        # E8/Leech lattice constraints
    CONSERVATION = "conservation"  # ΔΦ ≤ 0 constraints
    PARITY = "parity"             # Digital root constraints
    SYMMETRY = "symmetry"         # Weyl/symmetry constraints
    TOPOLOGICAL = "topological"   # Closure/continuity constraints


@dataclass
class Policy:
    """Represents a governance policy."""
    policy_id: str
    name: str
    description: str
    digital_root: int  # DR 0-9
    governance_level: GovernanceLevel
    constraint_types: List[ConstraintType]
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"Policy(DR{self.digital_root}: {self.name}, level={self.governance_level.value})"


@dataclass
class ViolationRecord:
    """Records a policy violation."""
    violation_id: str
    policy_id: str
    constraint_type: ConstraintType
    severity: str  # "error", "warning", "info"
    message: str
    timestamp: float
    resolved: bool = False
    
    def __repr__(self) -> str:
        status = "✓" if self.resolved else "✗"
        return f"Violation[{status}]: {self.severity.upper()} - {self.message}"


class PolicyHierarchy:
    """
    Hierarchical policy management system.
    
    Policies are organized by digital root (DR 0-9), with each DR
    corresponding to a specific governance level and set of constraints.
    
    DR 0: Ultimate governance (gravitational)
    DR 1-3: Permissive (electromagnetic)
    DR 4-6: Standard (weak/strong nuclear)
    DR 7-8: Strict (completion/infinity)
    DR 9: Ultimate (return to source)
    """
    
    def __init__(self):
        """Initialize policy hierarchy."""
        self.policies: Dict[str, Policy] = {}
        self.violations: List[ViolationRecord] = []
        self.active_policy_id: Optional[str] = None
        
        # Initialize built-in policies
        self._initialize_builtin_policies()
    
    def _initialize_builtin_policies(self):
        """Initialize the 10 built-in policies (DR 0-9)."""
        
        # DR 0: Gravitational governance (ultimate)
        self.register_policy(
            policy_id="dr0_gravitational",
            name="Gravitational Governance",
            description="Ultimate governance - all constraints active",
            digital_root=0,
            governance_level=GovernanceLevel.ULTIMATE,
            constraint_types=[
                ConstraintType.GEOMETRIC,
                ConstraintType.CONSERVATION,
                ConstraintType.PARITY,
                ConstraintType.SYMMETRY,
                ConstraintType.TOPOLOGICAL
            ]
        )
        
        # DR 1: Unity (permissive)
        self.register_policy(
            policy_id="dr1_unity",
            name="Unity Policy",
            description="Permissive governance - minimal constraints",
            digital_root=1,
            governance_level=GovernanceLevel.PERMISSIVE,
            constraint_types=[ConstraintType.GEOMETRIC]
        )
        
        # DR 2: Duality (permissive)
        self.register_policy(
            policy_id="dr2_duality",
            name="Duality Policy",
            description="Permissive governance - geometric + conservation",
            digital_root=2,
            governance_level=GovernanceLevel.PERMISSIVE,
            constraint_types=[
                ConstraintType.GEOMETRIC,
                ConstraintType.CONSERVATION
            ]
        )
        
        # DR 3: Trinity (permissive)
        self.register_policy(
            policy_id="dr3_trinity",
            name="Trinity Policy",
            description="Permissive governance - geometric + conservation + parity",
            digital_root=3,
            governance_level=GovernanceLevel.PERMISSIVE,
            constraint_types=[
                ConstraintType.GEOMETRIC,
                ConstraintType.CONSERVATION,
                ConstraintType.PARITY
            ]
        )
        
        # DR 4: Stability (standard)
        self.register_policy(
            policy_id="dr4_stability",
            name="Stability Policy",
            description="Standard governance - balanced constraints",
            digital_root=4,
            governance_level=GovernanceLevel.STANDARD,
            constraint_types=[
                ConstraintType.GEOMETRIC,
                ConstraintType.CONSERVATION,
                ConstraintType.PARITY,
                ConstraintType.SYMMETRY
            ]
        )
        
        # DR 5: Change (standard)
        self.register_policy(
            policy_id="dr5_change",
            name="Change Policy",
            description="Standard governance - dynamic constraints",
            digital_root=5,
            governance_level=GovernanceLevel.STANDARD,
            constraint_types=[
                ConstraintType.GEOMETRIC,
                ConstraintType.CONSERVATION,
                ConstraintType.PARITY
            ]
        )
        
        # DR 6: Harmony (standard)
        self.register_policy(
            policy_id="dr6_harmony",
            name="Harmony Policy",
            description="Standard governance - harmonic constraints",
            digital_root=6,
            governance_level=GovernanceLevel.STANDARD,
            constraint_types=[
                ConstraintType.GEOMETRIC,
                ConstraintType.CONSERVATION,
                ConstraintType.PARITY,
                ConstraintType.SYMMETRY
            ]
        )
        
        # DR 7: Completion (strict)
        self.register_policy(
            policy_id="dr7_completion",
            name="Completion Policy",
            description="Strict governance - enhanced validation",
            digital_root=7,
            governance_level=GovernanceLevel.STRICT,
            constraint_types=[
                ConstraintType.GEOMETRIC,
                ConstraintType.CONSERVATION,
                ConstraintType.PARITY,
                ConstraintType.SYMMETRY,
                ConstraintType.TOPOLOGICAL
            ]
        )
        
        # DR 8: Infinity (strict)
        self.register_policy(
            policy_id="dr8_infinity",
            name="Infinity Policy",
            description="Strict governance - infinite precision",
            digital_root=8,
            governance_level=GovernanceLevel.STRICT,
            constraint_types=[
                ConstraintType.GEOMETRIC,
                ConstraintType.CONSERVATION,
                ConstraintType.PARITY,
                ConstraintType.SYMMETRY,
                ConstraintType.TOPOLOGICAL
            ]
        )
        
        # DR 9: Return (ultimate)
        self.register_policy(
            policy_id="dr9_return",
            name="Return Policy",
            description="Ultimate governance - return to source",
            digital_root=9,
            governance_level=GovernanceLevel.ULTIMATE,
            constraint_types=[
                ConstraintType.GEOMETRIC,
                ConstraintType.CONSERVATION,
                ConstraintType.PARITY,
                ConstraintType.SYMMETRY,
                ConstraintType.TOPOLOGICAL
            ]
        )
        
        # Set default active policy to DR 4 (stability)
        self.active_policy_id = "dr4_stability"
    
    def register_policy(self,
                       policy_id: str,
                       name: str,
                       description: str,
                       digital_root: int,
                       governance_level: GovernanceLevel,
                       constraint_types: List[ConstraintType]) -> Policy:
        """
        Register a new policy.
        
        Args:
            policy_id: Unique policy identifier
            name: Human-readable policy name
            description: Policy description
            digital_root: Digital root (0-9)
            governance_level: Governance level
            constraint_types: List of constraint types
        
        Returns:
            Registered policy
        """
        if not 0 <= digital_root <= 9:
            raise ValueError(f"Digital root must be 0-9, got {digital_root}")
        
        policy = Policy(
            policy_id=policy_id,
            name=name,
            description=description,
            digital_root=digital_root,
            governance_level=governance_level,
            constraint_types=constraint_types
        )
        
        self.policies[policy_id] = policy
        return policy
    
    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get policy by ID."""
        return self.policies.get(policy_id)
    
    def get_policy_by_dr(self, digital_root: int) -> Optional[Policy]:
        """Get policy by digital root."""
        for policy in self.policies.values():
            if policy.digital_root == digital_root:
                return policy
        return None
    
    def set_active_policy(self, policy_id: str):
        """Set the active policy."""
        if policy_id not in self.policies:
            raise ValueError(f"Unknown policy: {policy_id}")
        
        self.active_policy_id = policy_id
    
    def get_active_policy(self) -> Optional[Policy]:
        """Get the currently active policy."""
        if self.active_policy_id:
            return self.policies[self.active_policy_id]
        return None
    
    def record_violation(self,
                        policy_id: str,
                        constraint_type: ConstraintType,
                        severity: str,
                        message: str) -> ViolationRecord:
        """
        Record a policy violation.
        
        Args:
            policy_id: Policy that was violated
            constraint_type: Type of constraint violated
            severity: Severity level ("error", "warning", "info")
            message: Violation message
        
        Returns:
            ViolationRecord
        """
        import time
        
        violation_id = f"v{len(self.violations):06d}"
        
        violation = ViolationRecord(
            violation_id=violation_id,
            policy_id=policy_id,
            constraint_type=constraint_type,
            severity=severity,
            message=message,
            timestamp=time.time()
        )
        
        self.violations.append(violation)
        return violation
    
    def get_violations(self, 
                      policy_id: Optional[str] = None,
                      resolved: Optional[bool] = None) -> List[ViolationRecord]:
        """
        Get violations, optionally filtered.
        
        Args:
            policy_id: Filter by policy ID
            resolved: Filter by resolution status
        
        Returns:
            List of matching violations
        """
        violations = self.violations
        
        if policy_id is not None:
            violations = [v for v in violations if v.policy_id == policy_id]
        
        if resolved is not None:
            violations = [v for v in violations if v.resolved == resolved]
        
        return violations
    
    def resolve_violation(self, violation_id: str):
        """Mark a violation as resolved."""
        for violation in self.violations:
            if violation.violation_id == violation_id:
                violation.resolved = True
                break
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of all policies."""
        return {
            "total_policies": len(self.policies),
            "active_policy": self.active_policy_id,
            "total_violations": len(self.violations),
            "unresolved_violations": len([v for v in self.violations if not v.resolved]),
            "policies_by_level": self._count_policies_by_level()
        }
    
    def _count_policies_by_level(self) -> Dict[str, int]:
        """Count policies by governance level."""
        counts = {}
        for policy in self.policies.values():
            level = policy.governance_level.value
            counts[level] = counts.get(level, 0) + 1
        return counts
    
    def __repr__(self) -> str:
        active = self.get_active_policy()
        active_name = active.name if active else "None"
        return f"PolicyHierarchy({len(self.policies)} policies, active: {active_name})"
