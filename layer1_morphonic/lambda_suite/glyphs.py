
from typing import Dict, Any
from . import ast as A

# Simple glyph table: maps runes to AST constructors
GLYPH_TABLE: Dict[str, Any] = {
    "λ": A.Lam,
    "•": A.App,
    "×": A.Pair,
    "→": "ARROW",  # used in types, not terms
    "μ": A.Mu,
    "⊳": A.Fst,
    "⊲": A.Snd,
}

def parse_tokens(tokens):
    # Minimal demo parser; real grammar would be EBNF.
    stack = []
    for t in tokens:
        if t == "true": stack.append(A.Const("true", True))
        elif t == "false": stack.append(A.Const("false", False))
        elif t.isdigit(): stack.append(A.Const("nat", int(t)))
        elif t == "pair":
            b = stack.pop(); a = stack.pop(); stack.append(A.Pair(a,b))
        elif t == "fst": stack.append(A.Fst(stack.pop()))
        elif t == "snd": stack.append(A.Snd(stack.pop()))
        else:
            stack.append(A.Var(t))
    return stack[-1] if stack else None
