
from typing import Any, Dict, Tuple
from . import ast as A

class EvalError(Exception): pass

def is_value(e):
    return isinstance(e, (A.Lam, A.Const, A.Pair))

def subst(body, var, val):
    # Very naive capture-avoiding substitution for demo purposes
    if isinstance(body, A.Var) and body.name == var: return val
    if isinstance(body, A.Lam) and body.var == var: return body
    if isinstance(body, A.Lam): return A.Lam(body.var, subst(body.body, var, val))
    if isinstance(body, A.App): return A.App(subst(body.fn, var, val), subst(body.arg, var, val))
    if isinstance(body, A.Let): return A.Let(body.var, subst(body.val, var, val), subst(body.body, var, val))
    if isinstance(body, A.Pair): return A.Pair(subst(body.fst, var, val), subst(body.snd, var, val))
    if isinstance(body, A.Fst): return A.Fst(subst(body.pair, var, val))
    if isinstance(body, A.Snd): return A.Snd(subst(body.pair, var, val))
    if isinstance(body, A.If): return A.If(subst(body.cond, var, val), subst(body.then, var, val), subst(body.els, var, val))
    if isinstance(body, A.Mu): return A.Mu(body.var, body.body if body.var==var else subst(body.body, var, val))
    return body

def delta_step(e):
    # Simple δ-reductions for constants
    if isinstance(e, A.App) and isinstance(e.fn, A.Const) and isinstance(e.arg, A.Const):
        if e.fn.name == "succ" and isinstance(e.arg.value, int):
            return A.Const("nat", e.arg.value + 1), True
        if e.fn.name == "pred" and isinstance(e.arg.value, int):
            return A.Const("nat", max(0, e.arg.value - 1)), True
        if e.fn.name == "iszero" and isinstance(e.arg.value, int):
            return A.Const("bool", e.arg.value == 0), True
    return e, False

def step(e):
    # β
    if isinstance(e, A.App) and isinstance(e.fn, A.Lam) and is_value(e.arg):
        return subst(e.fn.body, e.fn.var, e.arg), True
    # δ
    e2, did = delta_step(e)
    if did: return e2, True
    # structural
    if isinstance(e, A.App):
        if not is_value(e.fn):
            f2, d = step(e.fn)
            if d: return A.App(f2, e.arg), True
        if not is_value(e.arg):
            a2, d = step(e.arg)
            if d: return A.App(e.fn, a2), True
        return e, False
    if isinstance(e, A.Let):
        if is_value(e.val): return subst(e.body, e.var, e.val), True
        v2, d = step(e.val); 
        if d: return A.Let(e.var, v2, e.body), True
        return e, False
    if isinstance(e, A.Pair):
        if not is_value(e.fst):
            p, d = step(e.fst); 
            if d: return A.Pair(p, e.snd), True
        if not is_value(e.snd):
            q, d = step(e.snd);
            if d: return A.Pair(e.fst, q), True
        return e, False
    if isinstance(e, A.Fst) and isinstance(e.pair, A.Pair) and is_value(e.pair.fst):
        return e.pair.fst, True
    if isinstance(e, A.Snd) and isinstance(e.pair, A.Pair) and is_value(e.pair.snd):
        return e.pair.snd, True
    if isinstance(e, A.If) and isinstance(e.cond, A.Const) and isinstance(e.cond.value, bool):
        return (e.then if e.cond.value else e.els), True
    if isinstance(e, A.Mu):
        # μx.body -> body[x := μx.body]
        return subst(e.body, e.var, e), True
    return e, False

def eval_normal(e, fuel=10_000):
    steps = 0
    while steps < fuel:
        e2, did = step(e)
        if not did: return e, steps
        e = e2; steps += 1
    raise EvalError("Out of fuel")
