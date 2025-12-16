#!/usr/bin/env python3
"""
CQE Governance Engine
Universal constraint management and validation using CQE principles
"""

import numpy as np
import time
from typing import Any, Dict, List, Tuple, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import json
import hashlib

from ..core.cqe_os_kernel import CQEAtom, CQEKernel, CQEOperationType

class GovernanceLevel(Enum):
    """Levels of governance enforcement"""
    PERMISSIVE = "permissive"      # Minimal constraints
    STANDARD = "standard"          # Normal CQE constraints
    STRICT = "strict"              # Enhanced validation
    TQF_LAWFUL = "tqf_lawful"     # TQF quaternary constraints
    UVIBS_COMPLIANT = "uvibs_compliant"  # UVIBS Monster group constraints
    ULTIMATE = "ultimate"          # All constraints active

class ConstraintType(Enum):
    """Types of constraints in CQE governance"""
    QUAD_CONSTRAINT = "quad_constraint"
    E8_CONSTRAINT = "e8_constraint"
    PARITY_CONSTRAINT = "parity_constraint"
    GOVERNANCE_CONSTRAINT = "governance_constraint"
    TEMPORAL_CONSTRAINT = "temporal_constraint"
    SPATIAL_CONSTRAINT = "spatial_constraint"
    LOGICAL_CONSTRAINT = "logical_constraint"
    SEMANTIC_CONSTRAINT = "semantic_constraint"

@dataclass
class CQEConstraint:
    """Represents a constraint in CQE governance"""
    constraint_id: str
    constraint_type: ConstraintType
    name: str
    description: str
    validation_function: Callable[[CQEAtom], bool]
    repair_function: Optional[Callable[[CQEAtom], CQEAtom]] = None
    severity: str = "error"  # error, warning, info
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GovernancePolicy:
    """Represents a governance policy"""
    policy_id: str
    name: str
    description: str
    governance_level: GovernanceLevel
    constraints: List[str]  # Constraint IDs
    enforcement_rules: Dict[str, Any]
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ViolationRecord:
    """Records a governance violation"""
    violation_id: str
    atom_id: str
    constraint_id: str
    violation_type: str
    severity: str
    timestamp: float
    details: Dict[str, Any]
    resolved: bool = False
    resolution_method: Optional[str] = None

class CQEGovernanceEngine:
    """Universal governance engine using CQE principles"""
    
    def __init__(self, kernel: CQEKernel):
        self.kernel = kernel
        self.constraints: Dict[str, CQEConstraint] = {}
        self.policies: Dict[str, GovernancePolicy] = {}
        self.violations: Dict[str, ViolationRecord] = {}
        self.active_policy: Optional[str] = None
        self.governance_level = GovernanceLevel.STANDARD
        
        # Governance state
        self.enforcement_active = True
        self.auto_repair = True
        self.violation_threshold = 10
        
        # Monitoring
        self.violation_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        
        # Threading
        self.governance_lock = threading.RLock()
        
        # Initialize built-in constraints and policies
        self._initialize_builtin_constraints()
        self._initialize_builtin_policies()
    
    def _initialize_builtin_constraints(self):
        """Initialize built-in CQE constraints"""
        
        # Quad Constraints
        self.register_constraint(
            constraint_type=ConstraintType.QUAD_CONSTRAINT,
            name="valid_quad_range",
            description="Quad values must be in range [1,4]",
            validation_function=lambda atom: all(1 <= q <= 4 for q in atom.quad_encoding),
            repair_function=self._repair_quad_range
        )
        
        self.register_constraint(
            constraint_type=ConstraintType.QUAD_CONSTRAINT,
            name="quad_palindrome_symmetry",
            description="Quad encoding should exhibit palindromic properties",
            validation_function=self._validate_quad_palindrome,
            repair_function=self._repair_quad_palindrome,
            severity="warning"
        )
        
        # E8 Constraints
        self.register_constraint(
            constraint_type=ConstraintType.E8_CONSTRAINT,
            name="e8_lattice_membership",
            description="E8 embedding must be valid lattice point",
            validation_function=self._validate_e8_lattice,
            repair_function=self._repair_e8_lattice
        )
        
        self.register_constraint(
            constraint_type=ConstraintType.E8_CONSTRAINT,
            name="e8_norm_bounds",
            description="E8 embedding norm must be within reasonable bounds",
            validation_function=lambda atom: 0.1 <= np.linalg.norm(atom.e8_embedding) <= 5.0,
            repair_function=self._repair_e8_norm
        )
        
        # Parity Constraints
        self.register_constraint(
            constraint_type=ConstraintType.PARITY_CONSTRAINT,
            name="parity_channel_consistency",
            description="Parity channels must be consistent with quad encoding",
            validation_function=self._validate_parity_consistency,
            repair_function=self._repair_parity_consistency
        )
        
        self.register_constraint(
            constraint_type=ConstraintType.PARITY_CONSTRAINT,
            name="golay_code_compliance",
            description="Parity channels should follow Golay code principles",
            validation_function=self._validate_golay_compliance,
            repair_function=self._repair_golay_compliance,
            severity="warning"
        )
        
        # Governance Constraints
        self.register_constraint(
            constraint_type=ConstraintType.GOVERNANCE_CONSTRAINT,
            name="lawful_state_requirement",
            description="Atoms must maintain lawful governance state",
            validation_function=lambda atom: atom.governance_state != "unlawful",
            repair_function=self._repair_governance_state
        )
        
        self.register_constraint(
            constraint_type=ConstraintType.GOVERNANCE_CONSTRAINT,
            name="tqf_orbit4_symmetry",
            description="TQF atoms must satisfy Orbit4 symmetry requirements",
            validation_function=self._validate_tqf_symmetry,
            repair_function=self._repair_tqf_symmetry
        )
        
        # Temporal Constraints
        self.register_constraint(
            constraint_type=ConstraintType.TEMPORAL_CONSTRAINT,
            name="timestamp_validity",
            description="Timestamps must be valid and recent",
            validation_function=self._validate_timestamp,
            repair_function=self._repair_timestamp,
            severity="warning"
        )
        
        # Spatial Constraints
        self.register_constraint(
            constraint_type=ConstraintType.SPATIAL_CONSTRAINT,
            name="spatial_locality",
            description="Related atoms should be spatially close in E8 space",
            validation_function=self._validate_spatial_locality,
            repair_function=self._repair_spatial_locality,
            severity="info"
        )
        
        # Logical Constraints
        self.register_constraint(
            constraint_type=ConstraintType.LOGICAL_CONSTRAINT,
            name="logical_consistency",
            description="Atom data must be logically consistent",
            validation_function=self._validate_logical_consistency,
            repair_function=self._repair_logical_consistency
        )
        
        # Semantic Constraints
        self.register_constraint(
            constraint_type=ConstraintType.SEMANTIC_CONSTRAINT,
            name="semantic_coherence",
            description="Atom data must be semantically coherent",
            validation_function=self._validate_semantic_coherence,
            repair_function=self._repair_semantic_coherence,
            severity="warning"
        )
    
    def _initialize_builtin_policies(self):
        """Initialize built-in governance policies"""
        
        # Permissive Policy
        self.register_policy(
            name="permissive",
            description="Minimal constraints for maximum flexibility",
            governance_level=GovernanceLevel.PERMISSIVE,
            constraints=[
                "valid_quad_range",
                "lawful_state_requirement"
            ],
            enforcement_rules={
                "auto_repair": True,
                "violation_threshold": 100,
                "strict_enforcement": False
            }
        )
        
        # Standard Policy
        self.register_policy(
            name="standard",
            description="Standard CQE governance with balanced constraints",
            governance_level=GovernanceLevel.STANDARD,
            constraints=[
                "valid_quad_range",
                "e8_lattice_membership",
                "e8_norm_bounds",
                "parity_channel_consistency",
                "lawful_state_requirement",
                "timestamp_validity"
            ],
            enforcement_rules={
                "auto_repair": True,
                "violation_threshold": 50,
                "strict_enforcement": True
            }
        )
        
        # Strict Policy
        self.register_policy(
            name="strict",
            description="Enhanced validation with strict constraints",
            governance_level=GovernanceLevel.STRICT,
            constraints=[
                "valid_quad_range",
                "quad_palindrome_symmetry",
                "e8_lattice_membership",
                "e8_norm_bounds",
                "parity_channel_consistency",
                "golay_code_compliance",
                "lawful_state_requirement",
                "timestamp_validity",
                "logical_consistency"
            ],
            enforcement_rules={
                "auto_repair": True,
                "violation_threshold": 20,
                "strict_enforcement": True
            }
        )
        
        # TQF Lawful Policy
        self.register_policy(
            name="tqf_lawful",
            description="TQF quaternary governance with Orbit4 symmetries",
            governance_level=GovernanceLevel.TQF_LAWFUL,
            constraints=[
                "valid_quad_range",
                "quad_palindrome_symmetry",
                "e8_lattice_membership",
                "parity_channel_consistency",
                "tqf_orbit4_symmetry",
                "timestamp_validity",
                "logical_consistency",
                "semantic_coherence"
            ],
            enforcement_rules={
                "auto_repair": True,
                "violation_threshold": 10,
                "strict_enforcement": True,
                "tqf_specific": True
            }
        )
        
        # UVIBS Compliant Policy
        self.register_policy(
            name="uvibs_compliant",
            description="UVIBS Monster group governance with 80D constraints",
            governance_level=GovernanceLevel.UVIBS_COMPLIANT,
            constraints=[
                "valid_quad_range",
                "e8_lattice_membership",
                "e8_norm_bounds",
                "parity_channel_consistency",
                "golay_code_compliance",
                "lawful_state_requirement",
                "spatial_locality",
                "logical_consistency"
            ],
            enforcement_rules={
                "auto_repair": True,
                "violation_threshold": 5,
                "strict_enforcement": True,
                "uvibs_specific": True
            }
        )
        
        # Ultimate Policy
        self.register_policy(
            name="ultimate",
            description="All constraints active with maximum governance",
            governance_level=GovernanceLevel.ULTIMATE,
            constraints=list(self.constraints.keys()),
            enforcement_rules={
                "auto_repair": True,
                "violation_threshold": 1,
                "strict_enforcement": True,
                "ultimate_mode": True
            }
        )
        
        # Set default policy
        self.set_active_policy("standard")
    
    def register_constraint(self, constraint_type: ConstraintType, name: str,
                          description: str, validation_function: Callable[[CQEAtom], bool],
                          repair_function: Optional[Callable[[CQEAtom], CQEAtom]] = None,
                          severity: str = "error", metadata: Dict[str, Any] = None) -> str:
        """Register a new constraint"""
        constraint_id = hashlib.md5(f"{constraint_type.value}:{name}".encode()).hexdigest()
        
        constraint = CQEConstraint(
            constraint_id=constraint_id,
            constraint_type=constraint_type,
            name=name,
            description=description,
            validation_function=validation_function,
            repair_function=repair_function,
            severity=severity,
            metadata=metadata or {}
        )
        
        with self.governance_lock:
            self.constraints[constraint_id] = constraint
        
        return constraint_id
    
    def register_policy(self, name: str, description: str, governance_level: GovernanceLevel,
                       constraints: List[str], enforcement_rules: Dict[str, Any],
                       metadata: Dict[str, Any] = None) -> str:
        """Register a new governance policy"""
        policy_id = hashlib.md5(f"{governance_level.value}:{name}".encode()).hexdigest()
        
        policy = GovernancePolicy(
            policy_id=policy_id,
            name=name,
            description=description,
            governance_level=governance_level,
            constraints=constraints,
            enforcement_rules=enforcement_rules,
            metadata=metadata or {}
        )
        
        with self.governance_lock:
            self.policies[policy_id] = policy
        
        return policy_id
    
    def set_active_policy(self, policy_name: str) -> bool:
        """Set the active governance policy"""
        with self.governance_lock:
            for policy_id, policy in self.policies.items():
                if policy.name == policy_name:
                    self.active_policy = policy_id
                    self.governance_level = policy.governance_level
                    
                    # Update enforcement settings
                    rules = policy.enforcement_rules
                    self.auto_repair = rules.get("auto_repair", True)
                    self.violation_threshold = rules.get("violation_threshold", 10)
                    self.enforcement_active = rules.get("strict_enforcement", True)
                    
                    return True
        
        return False
    
    def validate_atom(self, atom: CQEAtom) -> Tuple[bool, List[ViolationRecord]]:
        """Validate atom against active governance policy"""
        if not self.enforcement_active or not self.active_policy:
            return True, []
        
        with self.governance_lock:
            policy = self.policies[self.active_policy]
            violations = []
            
            for constraint_id in policy.constraints:
                if constraint_id not in self.constraints:
                    continue
                
                constraint = self.constraints[constraint_id]
                if not constraint.active:
                    continue
                
                try:
                    is_valid = constraint.validation_function(atom)
                    
                    if not is_valid:
                        violation = ViolationRecord(
                            violation_id=f"{atom.id}:{constraint_id}:{time.time()}",
                            atom_id=atom.id,
                            constraint_id=constraint_id,
                            violation_type=constraint.constraint_type.value,
                            severity=constraint.severity,
                            timestamp=time.time(),
                            details={
                                "constraint_name": constraint.name,
                                "constraint_description": constraint.description,
                                "atom_data": str(atom.data)[:100]  # Truncated for storage
                            }
                        )
                        
                        violations.append(violation)
                        self.violations[violation.violation_id] = violation
                        self.violation_history.append(violation.violation_id)
                
                except Exception as e:
                    # Create error violation
                    error_violation = ViolationRecord(
                        violation_id=f"{atom.id}:error:{time.time()}",
                        atom_id=atom.id,
                        constraint_id=constraint_id,
                        violation_type="validation_error",
                        severity="error",
                        timestamp=time.time(),
                        details={
                            "error": str(e),
                            "constraint_name": constraint.name
                        }
                    )
                    violations.append(error_violation)
            
            is_valid = len(violations) == 0
            return is_valid, violations
    
    def repair_atom(self, atom: CQEAtom, violations: List[ViolationRecord] = None) -> CQEAtom:
        """Repair atom violations using constraint repair functions"""
        if not self.auto_repair:
            return atom
        
        if violations is None:
            _, violations = self.validate_atom(atom)
        
        repaired_atom = atom
        
        with self.governance_lock:
            for violation in violations:
                constraint_id = violation.constraint_id
                
                if constraint_id not in self.constraints:
                    continue
                
                constraint = self.constraints[constraint_id]
                
                if constraint.repair_function:
                    try:
                        repaired_atom = constraint.repair_function(repaired_atom)
                        
                        # Mark violation as resolved
                        violation.resolved = True
                        violation.resolution_method = f"auto_repair:{constraint.name}"
                        
                    except Exception as e:
                        # Log repair failure
                        violation.details["repair_error"] = str(e)
        
        return repaired_atom
    
    def enforce_governance(self, atom_ids: List[str]) -> Dict[str, Any]:
        """Enforce governance on a set of atoms"""
        results = {
            "validated": 0,
            "violations": 0,
            "repaired": 0,
            "failed": 0,
            "violation_details": []
        }
        
        for atom_id in atom_ids:
            atom = self.kernel.memory_manager.retrieve_atom(atom_id)
            if not atom:
                results["failed"] += 1
                continue
            
            # Validate atom
            is_valid, violations = self.validate_atom(atom)
            results["validated"] += 1
            
            if violations:
                results["violations"] += len(violations)
                results["violation_details"].extend([v.violation_id for v in violations])
                
                # Repair if enabled
                if self.auto_repair:
                    repaired_atom = self.repair_atom(atom, violations)
                    
                    # Update atom in memory
                    self.kernel.memory_manager.store_atom(repaired_atom)
                    results["repaired"] += 1
        
        return results
    
    def get_governance_status(self) -> Dict[str, Any]:
        """Get comprehensive governance status"""
        with self.governance_lock:
            active_policy_info = None
            if self.active_policy:
                policy = self.policies[self.active_policy]
                active_policy_info = {
                    "name": policy.name,
                    "level": policy.governance_level.value,
                    "constraints": len(policy.constraints),
                    "enforcement_rules": policy.enforcement_rules
                }
            
            recent_violations = list(self.violation_history)[-10:]  # Last 10 violations
            
            violation_stats = {
                "total": len(self.violations),
                "resolved": sum(1 for v in self.violations.values() if v.resolved),
                "by_severity": defaultdict(int),
                "by_type": defaultdict(int)
            }
            
            for violation in self.violations.values():
                violation_stats["by_severity"][violation.severity] += 1
                violation_stats["by_type"][violation.violation_type] += 1
            
            return {
                "enforcement_active": self.enforcement_active,
                "auto_repair": self.auto_repair,
                "governance_level": self.governance_level.value,
                "active_policy": active_policy_info,
                "constraints": {
                    "total": len(self.constraints),
                    "active": sum(1 for c in self.constraints.values() if c.active),
                    "by_type": {ct.value: sum(1 for c in self.constraints.values() 
                                            if c.constraint_type == ct) 
                               for ct in ConstraintType}
                },
                "policies": {
                    "total": len(self.policies),
                    "active": sum(1 for p in self.policies.values() if p.active)
                },
                "violations": violation_stats,
                "recent_violations": recent_violations
            }
    
    # Constraint Validation Functions
    def _validate_quad_palindrome(self, atom: CQEAtom) -> bool:
        """Validate quad palindromic properties"""
        q1, q2, q3, q4 = atom.quad_encoding
        # Check for palindromic or symmetric patterns
        return (q1 == q4 and q2 == q3) or (q1 + q4 == q2 + q3)
    
    def _validate_e8_lattice(self, atom: CQEAtom) -> bool:
        """Validate E8 lattice membership"""
        # Check if embedding is close to a valid E8 lattice point
        embedding = atom.e8_embedding
        
        # Check coordinate sum constraint (simplified)
        coord_sum = np.sum(embedding)
        return abs(coord_sum - round(coord_sum)) < 0.1
    
    def _validate_parity_consistency(self, atom: CQEAtom) -> bool:
        """Validate parity channel consistency"""
        q1, q2, q3, q4 = atom.quad_encoding
        expected_parity = [
            q1 % 2, q2 % 2, q3 % 2, q4 % 2,
            (q1 + q2) % 2, (q3 + q4) % 2,
            (q1 + q3) % 2, (q2 + q4) % 2
        ]
        
        return atom.parity_channels == expected_parity
    
    def _validate_golay_compliance(self, atom: CQEAtom) -> bool:
        """Validate Golay code compliance"""
        # Simplified Golay code check
        parity_sum = sum(atom.parity_channels)
        return parity_sum % 2 == 0  # Even parity
    
    def _validate_tqf_symmetry(self, atom: CQEAtom) -> bool:
        """Validate TQF Orbit4 symmetry"""
        if atom.governance_state != "tqf_lawful":
            return True  # Only applies to TQF atoms
        
        q1, q2, q3, q4 = atom.quad_encoding
        # TQF orbit4 symmetry check
        orbit_sum = (q1 + q2 + q3 + q4) % 4
        mirror_check = (q1 + q4) % 2 == (q2 + q3) % 2
        return orbit_sum == 0 and mirror_check
    
    def _validate_timestamp(self, atom: CQEAtom) -> bool:
        """Validate timestamp"""
        current_time = time.time()
        # Check if timestamp is reasonable (not too old or in future)
        return (current_time - 86400) <= atom.timestamp <= (current_time + 3600)
    
    def _validate_spatial_locality(self, atom: CQEAtom) -> bool:
        """Validate spatial locality in E8 space"""
        # Check if atom is reasonably close to related atoms
        if atom.parent_id:
            parent = self.kernel.memory_manager.retrieve_atom(atom.parent_id)
            if parent:
                distance = np.linalg.norm(atom.e8_embedding - parent.e8_embedding)
                return distance <= 3.0  # Reasonable distance threshold
        
        return True  # No parent to check against
    
    def _validate_logical_consistency(self, atom: CQEAtom) -> bool:
        """Validate logical consistency"""
        # Basic logical consistency checks
        if isinstance(atom.data, dict):
            # Check for contradictory boolean values
            bool_values = {k: v for k, v in atom.data.items() if isinstance(v, bool)}
            if len(bool_values) >= 2:
                # Simple contradiction check
                return not (True in bool_values.values() and False in bool_values.values())
        
        return True
    
    def _validate_semantic_coherence(self, atom: CQEAtom) -> bool:
        """Validate semantic coherence"""
        # Basic semantic coherence checks
        if isinstance(atom.data, str):
            # Check for reasonable string length and content
            return 0 < len(atom.data) <= 10000 and atom.data.isprintable()
        
        return True
    
    # Constraint Repair Functions
    def _repair_quad_range(self, atom: CQEAtom) -> CQEAtom:
        """Repair quad range violations"""
        q1, q2, q3, q4 = atom.quad_encoding
        repaired_quad = tuple(max(1, min(4, q)) for q in atom.quad_encoding)
        
        repaired_atom = CQEAtom(
            data=atom.data,
            quad_encoding=repaired_quad,
            parent_id=atom.id,
            metadata={**atom.metadata, "repaired": "quad_range"}
        )
        
        return repaired_atom
    
    def _repair_quad_palindrome(self, atom: CQEAtom) -> CQEAtom:
        """Repair quad palindrome violations"""
        q1, q2, q3, q4 = atom.quad_encoding
        
        # Create palindromic pattern
        avg_outer = (q1 + q4) // 2
        avg_inner = (q2 + q3) // 2
        
        repaired_quad = (avg_outer, avg_inner, avg_inner, avg_outer)
        
        repaired_atom = CQEAtom(
            data=atom.data,
            quad_encoding=repaired_quad,
            parent_id=atom.id,
            metadata={**atom.metadata, "repaired": "quad_palindrome"}
        )
        
        return repaired_atom
    
    def _repair_e8_lattice(self, atom: CQEAtom) -> CQEAtom:
        """Repair E8 lattice violations"""
        repaired_atom = CQEAtom(
            data=atom.data,
            quad_encoding=atom.quad_encoding,
            parent_id=atom.id,
            metadata={**atom.metadata, "repaired": "e8_lattice"}
        )
        
        # Re-project to E8 lattice
        repaired_atom._compute_e8_embedding()
        
        return repaired_atom
    
    def _repair_e8_norm(self, atom: CQEAtom) -> CQEAtom:
        """Repair E8 norm violations"""
        current_norm = np.linalg.norm(atom.e8_embedding)
        
        if current_norm > 5.0:
            # Scale down
            scale_factor = 4.0 / current_norm
            new_embedding = atom.e8_embedding * scale_factor
        elif current_norm < 0.1:
            # Scale up
            new_embedding = atom.e8_embedding * 10.0
        else:
            new_embedding = atom.e8_embedding
        
        repaired_atom = CQEAtom(
            data=atom.data,
            quad_encoding=atom.quad_encoding,
            parent_id=atom.id,
            metadata={**atom.metadata, "repaired": "e8_norm"}
        )
        
        repaired_atom.e8_embedding = repaired_atom._project_to_e8_lattice(new_embedding)
        
        return repaired_atom
    
    def _repair_parity_consistency(self, atom: CQEAtom) -> CQEAtom:
        """Repair parity consistency violations"""
        repaired_atom = CQEAtom(
            data=atom.data,
            quad_encoding=atom.quad_encoding,
            parent_id=atom.id,
            metadata={**atom.metadata, "repaired": "parity_consistency"}
        )
        
        # Recompute parity channels
        repaired_atom._compute_parity_channels()
        
        return repaired_atom
    
    def _repair_golay_compliance(self, atom: CQEAtom) -> CQEAtom:
        """Repair Golay code violations"""
        parity_channels = atom.parity_channels.copy()
        
        # Ensure even parity
        if sum(parity_channels) % 2 != 0:
            # Flip the last bit to achieve even parity
            parity_channels[-1] = 1 - parity_channels[-1]
        
        repaired_atom = CQEAtom(
            data=atom.data,
            quad_encoding=atom.quad_encoding,
            parent_id=atom.id,
            metadata={**atom.metadata, "repaired": "golay_compliance"}
        )
        
        repaired_atom.parity_channels = parity_channels
        
        return repaired_atom
    
    def _repair_governance_state(self, atom: CQEAtom) -> CQEAtom:
        """Repair governance state violations"""
        repaired_atom = CQEAtom(
            data=atom.data,
            quad_encoding=atom.quad_encoding,
            parent_id=atom.id,
            metadata={**atom.metadata, "repaired": "governance_state"}
        )
        
        # Re-validate governance
        repaired_atom._validate_governance()
        
        return repaired_atom
    
    def _repair_tqf_symmetry(self, atom: CQEAtom) -> CQEAtom:
        """Repair TQF symmetry violations"""
        q1, q2, q3, q4 = atom.quad_encoding
        
        # Adjust to satisfy TQF constraints
        # Ensure orbit sum is 0 mod 4
        current_sum = (q1 + q2 + q3 + q4) % 4
        if current_sum != 0:
            adjustment = (4 - current_sum) % 4
            q4 = ((q4 + adjustment - 1) % 4) + 1
        
        # Ensure mirror symmetry
        if (q1 + q4) % 2 != (q2 + q3) % 2:
            q4 = ((q4 + 1 - 1) % 4) + 1  # Adjust q4 by 1
        
        repaired_atom = CQEAtom(
            data=atom.data,
            quad_encoding=(q1, q2, q3, q4),
            parent_id=atom.id,
            metadata={**atom.metadata, "repaired": "tqf_symmetry"}
        )
        
        return repaired_atom
    
    def _repair_timestamp(self, atom: CQEAtom) -> CQEAtom:
        """Repair timestamp violations"""
        repaired_atom = CQEAtom(
            data=atom.data,
            quad_encoding=atom.quad_encoding,
            parent_id=atom.id,
            metadata={**atom.metadata, "repaired": "timestamp"}
        )
        
        # Update to current time
        repaired_atom.timestamp = time.time()
        
        return repaired_atom
    
    def _repair_spatial_locality(self, atom: CQEAtom) -> CQEAtom:
        """Repair spatial locality violations"""
        if atom.parent_id:
            parent = self.kernel.memory_manager.retrieve_atom(atom.parent_id)
            if parent:
                # Move closer to parent in E8 space
                direction = parent.e8_embedding - atom.e8_embedding
                distance = np.linalg.norm(direction)
                
                if distance > 3.0:
                    # Move to within acceptable distance
                    unit_direction = direction / distance
                    new_embedding = parent.e8_embedding - unit_direction * 2.5
                    
                    repaired_atom = CQEAtom(
                        data=atom.data,
                        quad_encoding=atom.quad_encoding,
                        parent_id=atom.id,
                        metadata={**atom.metadata, "repaired": "spatial_locality"}
                    )
                    
                    repaired_atom.e8_embedding = repaired_atom._project_to_e8_lattice(new_embedding)
                    
                    return repaired_atom
        
        return atom  # No repair needed or possible
    
    def _repair_logical_consistency(self, atom: CQEAtom) -> CQEAtom:
        """Repair logical consistency violations"""
        if isinstance(atom.data, dict):
            repaired_data = atom.data.copy()
            
            # Remove contradictory boolean values
            bool_keys = [k for k, v in repaired_data.items() if isinstance(v, bool)]
            if len(bool_keys) >= 2:
                # Keep only the first boolean value
                for key in bool_keys[1:]:
                    del repaired_data[key]
            
            repaired_atom = CQEAtom(
                data=repaired_data,
                quad_encoding=atom.quad_encoding,
                parent_id=atom.id,
                metadata={**atom.metadata, "repaired": "logical_consistency"}
            )
            
            return repaired_atom
        
        return atom
    
    def _repair_semantic_coherence(self, atom: CQEAtom) -> CQEAtom:
        """Repair semantic coherence violations"""
        if isinstance(atom.data, str):
            repaired_data = atom.data
            
            # Truncate if too long
            if len(repaired_data) > 10000:
                repaired_data = repaired_data[:10000]
            
            # Remove non-printable characters
            repaired_data = ''.join(c for c in repaired_data if c.isprintable())
            
            repaired_atom = CQEAtom(
                data=repaired_data,
                quad_encoding=atom.quad_encoding,
                parent_id=atom.id,
                metadata={**atom.metadata, "repaired": "semantic_coherence"}
            )
            
            return repaired_atom
        
        return atom

# Export main classes
__all__ = [
    'CQEGovernanceEngine', 'CQEConstraint', 'GovernancePolicy', 'ViolationRecord',
    'GovernanceLevel', 'ConstraintType'
]
