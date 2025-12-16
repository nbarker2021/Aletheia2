# API Reference (v0)

## Core
- `SNAPCore(storage).register_kind(kind, validator)`
- `SNAPCore.save(key, obj: dict) -> key`
- `SNAPCore.load(key: str) -> dict|None`
- `SNAPCore.query(kind: str=None, prefix: str=None) -> list[str]`

## Types (validators)
Register via `register_basic_types(snap)`:
- `ArXivPaper`: {kind,id,title,url,...}
- `PaperDigest`: {kind,paper_id,...}
- `LLMProfile`: {kind,id,context,tools,evals,...}
- `AlgorithmCard`: {kind,name,...}
- `CodeUnit` (SNAPBIT): {kind,id,lang,signature,body,deps,tests,...}
- `ErrorTrace`: {kind,id,exception,stack?,tags?,repro?}
- `PlaybookStep`: {kind,id,when[],do[],verify[]}
- `AnyFile`: {kind,path|bytes?,mimetype?,sha256?}

## SNAPDNA
Register via `register_snapdna(snap)`:
- `persona_create(snap, *, domain, level, sources, weights, notes) -> id`
- `persona_update_score(snap, id, delta, ema=0.6) -> score`
- `persona_compose(snap, hybrid_id, components=[(id, weight), ...]) -> id`

## Code stitcher
- `stitch(snap, goal_signature: dict, constraints: dict={}) -> CodeUnit`

## Playbook
- `apply_playbook(snap, error_key: str) -> dict`

## MoE (routing)
- `route(snap, task: {tags[], tools[]}, k=2) -> list[(persona_id, score)]`
