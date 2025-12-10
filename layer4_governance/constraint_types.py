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