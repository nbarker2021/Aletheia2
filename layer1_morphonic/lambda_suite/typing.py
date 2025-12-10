
from typing import Dict, Optional, Tuple
from . import ast as A
from .typesys import Type, Arrow, Bool, Nat, Prod, pretty

class TypeError_(Exception): pass

Env = Dict[str, Type]

def type_of(e: object, env: Env) -> Type:
    if isinstance(e, A.Var):
        if e.name not in env: raise TypeError_(f"Unbound var: {e.name}")
        return env[e.name]
    if isinstance(e, A.Lam):
        # We require an annotation via env for parameter (convention)
        if e.var not in env: raise TypeError_(f"Missing type for param: {e.var}")
        return Arrow(env[e.var], type_of(e.body, env))
    if isinstance(e, A.App):
        tf = type_of(e.fn, env); ta = type_of(e.arg, env)
        if not isinstance(tf, Arrow): raise TypeError_("Function type expected")
        if tf.src != ta: raise TypeError_(f"Type mismatch: {pretty(tf.src)} vs {pretty(ta)}")
        return tf.dst
    if isinstance(e, A.Const):
        if e.name == "true" or e.name == "false": return Bool()
        if isinstance(e.value, int): return Nat()
        return Nat() if isinstance(e.value, int) else Bool()
    if isinstance(e, A.Pair):
        return Prod(type_of(e.fst, env), type_of(e.snd, env))
    if isinstance(e, A.Fst):
        pt = type_of(e.pair, env)
        if not isinstance(pt, Prod): raise TypeError_("fst on non-pair")
        return pt.fst
    if isinstance(e, A.Snd):
        pt = type_of(e.pair, env)
        if not isinstance(pt, Prod): raise TypeError_("snd on non-pair")
        return pt.snd
    if isinstance(e, A.If):
        tc = type_of(e.cond, env)
        if not isinstance(tc, Bool): raise TypeError_("if cond must be Bool")
        tt = type_of(e.then, env); te = type_of(e.els, env)
        if tt != te: raise TypeError_("branches must agree")
        return tt
    if isinstance(e, A.Let):
        tv = type_of(e.val, env)
        env2 = dict(env); env2[e.var] = tv
        return type_of(e.body, env2)
    if isinstance(e, A.Mu):
        # crude iso-recursive typing: assume var type known
        if e.var not in env: raise TypeError_(f"Missing type for μ var: {e.var}")
        return type_of(e.body, env)
    raise TypeError_(f"Unknown term: {e}")
