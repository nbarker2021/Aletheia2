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
