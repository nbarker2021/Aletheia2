class LanguagePattern:
    """Represents a language pattern in CQE space"""
    pattern_id: str
    language_type: LanguageType
    syntax_level: SyntaxLevel
    pattern: str
    description: str
    quad_signature: Tuple[int, int, int, int]
    e8_embedding: np.ndarray
    frequency: int = 0
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

