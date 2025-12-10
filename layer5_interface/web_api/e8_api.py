
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import numpy as np

from e8.construct import e8_roots, is_in_e8_union
from e8.lattice import E8Lattice
from e8.weyl import reflect_through_root
from e8.invariants import e8_invariants_quick

app = FastAPI(title="E8 Lattice API")
lat = E8Lattice()
ROOTS = e8_roots()

class Vector(BaseModel):
    x: List[float] = Field(..., description="8-D vector")

@app.get("/invariants")
def invariants():
    return e8_invariants_quick()

@app.post("/snap")
def snap(v: Vector):
    x = np.array(v.x, dtype=float)
    y = lat.snap(x)
    return {"nearest": y.tolist()}

@app.post("/reflect")
def reflect(v: Vector, root_index: int):
    x = np.array(v.x, dtype=float)
    a = ROOTS[root_index % len(ROOTS)]
    y = reflect_through_root(x, a)
    return {"reflected": y.tolist(), "root_index": root_index % len(ROOTS)}

@app.post("/is_member")
def is_member(v: Vector):
    x = np.array(v.x, dtype=float)
    return {"in_e8": bool(is_in_e8_union(x))}
